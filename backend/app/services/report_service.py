from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from lightrag.utils import logger

from raganything.export import (
    export_chat_history_to_pdf,
    export_current_answer_to_pdf,
    generate_structured_agent_report,
    generate_pdf_report_from_index,
)

from backend.app.schemas.document import DocumentRecord
from backend.app.services.query_service import (
    CORPUS_RELEVANCE_THRESHOLD,
    extract_candidate_filenames,
    normalize_to_messages,
)

REPORT_TYPE_KEYWORDS: dict[str, set[str]] = {
    "technical_paper": {
        "technical report",
        "paper",
        "objective",
        "method",
        "dataset",
        "formula",
        "results",
        "limitations",
        "findings",
    },
    "summary": {"summary", "overview", "main points", "summarize"},
    "table_figure_analysis": {
        "table",
        "figure",
        "chart",
        "accuracy",
        "roc",
        "auc",
        "mlp",
        "formula",
    },
    "visual_report": {"image", "picture", "dog", "cat", "visual", "scene"},
}

REPORT_AMBIGUITY_DELTA = 6.0
REPORT_PAPER_FILE_TYPES = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}
REPORT_IMAGE_FILE_TYPES = {".jpg", ".jpeg", ".png"}
TECHNICAL_PAPER_REPORT_TYPES = {"technical_paper", "table_figure_analysis"}
VISUAL_REQUEST_TERMS = {"image", "visual", "picture", "photo", "dog", "cat", "scene"}
CONTAMINATED_IMAGE_MARKERS = {
    "[pdf table",
    "table 2",
    "accuracy",
    "roc",
    "mlp",
    "tabnet",
    "references",
    "citation",
    "dataset",
    "method",
    "results",
}


def parse_doc_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.match(r"^(.*)\s\[[0-9a-f]+\]$", str(value).strip())
    if m:
        return m.group(1).strip()
    return str(value).strip() or None


def parse_doc_id_from_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.search(r"\[([0-9a-f]+)\]$", str(value))
    return m.group(1) if m else None


def format_pdf_export_result(
    status: str, pdf_path: Optional[Path]
) -> tuple[str, Optional[str]]:
    return status, (str(pdf_path) if pdf_path else None)


def format_pdf_export_result_with_metadata(
    status: str,
    pdf_path: Optional[Path],
    metadata: Optional[dict] = None,
) -> tuple[str, Optional[str], dict]:
    return status, (str(pdf_path) if pdf_path else None), dict(metadata or {})


def infer_report_type(report_request: str) -> str:
    lowered = str(report_request or "").lower()
    # Keep inference deterministic so the UI can stay simple and the backend
    # remains explainable when the user asks why a report type was chosen.
    for report_type in [
        "technical_paper",
        "summary",
        "table_figure_analysis",
        "visual_report",
    ]:
        if any(keyword in lowered for keyword in REPORT_TYPE_KEYWORDS[report_type]):
            return report_type
    return "custom"


def infer_report_source_intent(report_request: str, report_type: str) -> str:
    lowered = str(report_request or "").lower()
    if report_type in TECHNICAL_PAPER_REPORT_TYPES:
        return "text_document"
    if report_type == "visual_report":
        return "visual"
    if any(term in lowered for term in VISUAL_REQUEST_TERMS):
        return "visual"
    if any(term in lowered for term in REPORT_TYPE_KEYWORDS["technical_paper"]):
        return "text_document"
    return "generic"


def _extract_report_terms(service, report_request: str) -> set[str]:
    if hasattr(service, "_extract_query_terms"):
        return set(service._extract_query_terms(report_request))
    return set(re.findall(r"[a-zA-Z0-9_]+", str(report_request or "").lower()))


def _is_report_text_document(file_type: str) -> bool:
    return str(file_type or "").lower() in REPORT_PAPER_FILE_TYPES


def _is_report_image_document(file_type: str) -> bool:
    return str(file_type or "").lower() in REPORT_IMAGE_FILE_TYPES


def _load_report_candidate_text(service, rec: DocumentRecord) -> dict[str, Any]:
    if hasattr(service, "_load_candidate_text_for_document"):
        return service._load_candidate_text_for_document(rec)
    return {
        "working_dir": str(getattr(rec, "working_dir_rel", "")),
        "texts": [],
        "previews": [],
    }


def _request_explicitly_targets_image(
    report_request: str, rec: DocumentRecord, report_terms: set[str]
) -> bool:
    filename = str(rec.original_filename or "").lower()
    stem = Path(filename).stem.lower()
    explicit_names = extract_candidate_filenames(report_request)
    if filename in explicit_names:
        return True
    return bool(
        report_terms.intersection(VISUAL_REQUEST_TERMS)
        or (stem and stem in str(report_request or "").lower())
    )


def _is_image_candidate_contaminated(
    *, candidate_texts: list[str], report_request: str, source_intent: str
) -> bool:
    if source_intent == "visual":
        return False
    text_blob = " ".join(candidate_texts[:8]).lower()
    request_lower = str(report_request or "").lower()
    if any(marker in request_lower for marker in VISUAL_REQUEST_TERMS):
        return False
    return any(marker in text_blob for marker in CONTAMINATED_IMAGE_MARKERS)


def _fallback_report_candidate_score(
    report_request: str,
    rec: DocumentRecord,
    *,
    selected_doc_id: Optional[str] = None,
) -> dict[str, Any]:
    lowered_request = str(report_request or "").lower()
    filename = str(rec.original_filename or "")
    filename_lower = filename.lower()
    stem_lower = Path(filename_lower).stem
    terms = set(re.findall(r"[a-zA-Z0-9_]+", lowered_request))
    filename_terms = {
        token for token in re.findall(r"[a-zA-Z0-9_]+", stem_lower) if len(token) >= 2
    }
    overlap = sorted(filename_terms.intersection(terms))
    score = (
        100.0
        if selected_doc_id and rec.doc_id == selected_doc_id
        else 35.0
        if filename_lower in lowered_request
        else 10.0
    )
    if overlap:
        score += float(len(overlap) * 12)
    return {
        "record": rec,
        "doc_id": rec.doc_id,
        "filename": filename,
        "score": score,
        "reason": (
            f"legacy fallback ranking; matched filename terms: {', '.join(overlap[:3])}"
            if overlap
            else "legacy fallback ranking"
        ),
        "unique_term_overlap": len(overlap),
        "file_type": Path(filename).suffix.lower(),
    }


def _same_report_source_family(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_type = str(first.get("file_type") or "").lower()
    second_type = str(second.get("file_type") or "").lower()
    if _is_report_text_document(first_type) and _is_report_text_document(second_type):
        return True
    if _is_report_image_document(first_type) and _is_report_image_document(second_type):
        return True
    return False


def _score_report_source_candidate(
    *,
    service,
    report_request: str,
    report_type: str,
    source_intent: str,
    rec: DocumentRecord,
    selected_doc_id: Optional[str] = None,
) -> dict[str, Any]:
    base_candidate = (
        service._score_document_candidate(report_request, rec)
        if hasattr(service, "_score_document_candidate")
        else _fallback_report_candidate_score(
            report_request, rec, selected_doc_id=selected_doc_id
        )
    )
    score = float(base_candidate["score"])
    reasons = [str(base_candidate.get("reason") or "")]
    report_terms = _extract_report_terms(service, report_request)
    file_type = str(base_candidate.get("file_type") or rec.file_type or "").lower()
    candidate_text = _load_report_candidate_text(service, rec)
    contaminated_for_request = _is_image_candidate_contaminated(
        candidate_texts=candidate_text.get("texts") or [],
        report_request=report_request,
        source_intent=source_intent,
    )
    explicit_image_request = _request_explicitly_targets_image(
        report_request, rec, report_terms
    )
    excluded = False
    exclusion_reason: Optional[str] = None
    intent_match = "generic"

    if source_intent == "text_document":
        intent_match = "paper" if _is_report_text_document(file_type) else "mismatch"
        if file_type == ".pdf":
            score += 50.0
            reasons.append("technical paper intent strongly prefers PDF files")
        elif _is_report_text_document(file_type):
            score += 35.0
            reasons.append("technical paper intent prefers text documents")
        elif _is_report_image_document(file_type):
            if contaminated_for_request and not explicit_image_request:
                excluded = True
                exclusion_reason = (
                    "excluded contaminated image candidate for technical paper request"
                )
                reasons.append(
                    "image candidate contained paper/table terms unrelated to a visual request"
                )
            elif not explicit_image_request:
                excluded = True
                exclusion_reason = (
                    "excluded image candidate for technical paper request"
                )
                reasons.append("technical paper intent ignores unrelated image files")
            else:
                score -= 40.0
                reasons.append(
                    "explicit image mention keeps image candidate, but penalized"
                )
    elif source_intent == "visual":
        intent_match = "visual" if _is_report_image_document(file_type) else "mismatch"
        if _is_report_image_document(file_type):
            score += 45.0
            reasons.append("visual report intent strongly prefers image files")
        elif _is_report_text_document(file_type):
            score -= 20.0
            reasons.append("visual report intent penalized text document")
    else:
        intent_match = (
            "text_document" if _is_report_text_document(file_type) else "visual"
        )

    if selected_doc_id and rec.doc_id == selected_doc_id:
        score += 100.0
        reasons.append("selected document id matched")

    request_lower = str(report_request or "").lower()
    filename = str(rec.original_filename or "")
    stem_lower = Path(filename.lower()).stem
    if filename.lower() in extract_candidate_filenames(report_request):
        score += 60.0
        reasons.append("explicit filename mention matched report request")
    elif stem_lower and stem_lower in request_lower:
        score += 25.0
        reasons.append("request mentioned filename stem")

    reason = "; ".join(part for part in reasons if part)[:600]
    if excluded:
        logger.warning(
            "Report source candidate excluded: doc_id=%s filename=%s file_type=%s report_type=%s source_intent=%s reason=%s",
            rec.doc_id,
            rec.original_filename,
            file_type,
            report_type,
            source_intent,
            exclusion_reason,
        )

    return {
        "doc_id": rec.doc_id,
        "filename": filename,
        "file_type": file_type,
        "score": score,
        "reason": reason or "no meaningful overlap",
        "intent_match": intent_match,
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
        "contaminated_for_request": contaminated_for_request,
        "unique_term_overlap": int(base_candidate.get("unique_term_overlap") or 0),
        "record": rec,
    }


def detect_report_source_candidates(
    *,
    service,
    report_request: str,
    selected_doc_id: Optional[str] = None,
) -> dict[str, Any]:
    inferred_report_type = infer_report_type(report_request)
    source_intent = infer_report_source_intent(report_request, inferred_report_type)
    if hasattr(service, "list_indexed_documents"):
        indexed_docs = service.list_indexed_documents(selected_doc_id=selected_doc_id)
    else:
        # Keep the refactor compatible with legacy Gradio helpers and minimal
        # test doubles that only expose list_documents().
        indexed_docs = [
            rec
            for rec in service.list_documents()
            if getattr(rec, "status", "") == "indexed"
            and (
                not selected_doc_id
                or str(getattr(rec, "doc_id", "")) == selected_doc_id
            )
        ]
    considered_documents: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    if not indexed_docs:
        return {
            "status": "no_source",
            "message": "Please upload and process a file before generating a PDF report.",
            "available_sources": [],
            "candidate_sources": [],
            "excluded_candidates": [],
            "selected_records": [],
            "considered_documents": considered_documents,
            "inferred_report_type": inferred_report_type,
            "source_intent": source_intent,
        }

    ranked = sorted(
        (
            _score_report_source_candidate(
                service=service,
                report_request=report_request,
                report_type=inferred_report_type,
                source_intent=source_intent,
                rec=rec,
                selected_doc_id=selected_doc_id,
            )
            for rec in indexed_docs
        ),
        key=lambda item: (
            item["score"],
            1 if item.get("file_type") == ".pdf" else 0,
            1 if item.get("file_type") in REPORT_PAPER_FILE_TYPES else 0,
            item.get("unique_term_overlap", 0),
            item["filename"],
        ),
        reverse=True,
    )
    considered_documents = [
        {
            "doc_id": item["doc_id"],
            "filename": item["filename"],
            "file_type": item["file_type"],
            "score": round(float(item["score"]), 4),
            "reason": item["reason"],
            "intent_match": item["intent_match"],
            "excluded": item["excluded"],
            "exclusion_reason": item["exclusion_reason"],
            "contaminated_for_request": item["contaminated_for_request"],
        }
        for item in ranked
    ]
    excluded_candidates = [
        {
            "doc_id": item["doc_id"],
            "filename": item["filename"],
            "file_type": item["file_type"],
            "score": round(float(item["score"]), 4),
            "reason": item["reason"],
            "intent_match": item["intent_match"],
            "excluded": True,
            "exclusion_reason": item["exclusion_reason"],
            "contaminated_for_request": item["contaminated_for_request"],
        }
        for item in ranked
        if item["excluded"]
    ]
    candidate_sources = [
        item
        for item in ranked
        if not item["excluded"] and item["score"] >= CORPUS_RELEVANCE_THRESHOLD
    ]
    if not candidate_sources:
        return {
            "status": "no_source",
            "message": "No relevant indexed source was found for this report request.",
            "available_sources": [rec.original_filename for rec in indexed_docs],
            "candidate_sources": [],
            "excluded_candidates": excluded_candidates,
            "selected_records": [],
            "considered_documents": considered_documents,
            "inferred_report_type": inferred_report_type,
            "source_intent": source_intent,
        }

    if selected_doc_id:
        return {
            "status": "selected",
            "message": None,
            "available_sources": [rec.original_filename for rec in indexed_docs],
            "candidate_sources": [
                item for item in considered_documents if not item["excluded"]
            ],
            "excluded_candidates": excluded_candidates,
            "selected_records": [candidate_sources[0]["record"]],
            "considered_documents": considered_documents,
            "inferred_report_type": inferred_report_type,
            "source_intent": source_intent,
        }

    compare_mode = (
        service._is_compare_question(report_request)
        if hasattr(service, "_is_compare_question")
        else False
    )
    if compare_mode:
        selected_records = [item["record"] for item in candidate_sources[:2]]
        return {
            "status": "selected",
            "message": None,
            "available_sources": [rec.original_filename for rec in indexed_docs],
            "candidate_sources": [
                item for item in considered_documents if not item["excluded"]
            ],
            "excluded_candidates": excluded_candidates,
            "selected_records": selected_records,
            "considered_documents": considered_documents,
            "inferred_report_type": inferred_report_type,
            "source_intent": source_intent,
        }

    if (
        len(candidate_sources) > 1
        and abs(
            float(candidate_sources[0]["score"]) - float(candidate_sources[1]["score"])
        )
        <= REPORT_AMBIGUITY_DELTA
        and _same_report_source_family(candidate_sources[0], candidate_sources[1])
    ):
        return {
            "status": "ambiguous",
            "message": "Multiple possible source documents were found. Please clarify the report request.",
            "available_sources": [rec.original_filename for rec in indexed_docs],
            "candidate_sources": [
                item for item in considered_documents if not item["excluded"]
            ][:3],
            "excluded_candidates": excluded_candidates,
            "selected_records": [],
            "considered_documents": considered_documents,
            "inferred_report_type": inferred_report_type,
            "source_intent": source_intent,
        }

    return {
        "status": "selected",
        "message": None,
        "available_sources": [rec.original_filename for rec in indexed_docs],
        "candidate_sources": [
            item for item in considered_documents if not item["excluded"]
        ],
        "excluded_candidates": excluded_candidates,
        "selected_records": [candidate_sources[0]["record"]],
        "considered_documents": considered_documents,
        "inferred_report_type": inferred_report_type,
        "source_intent": source_intent,
    }


def export_current_answer_action(
    question: str,
    answer: str,
    source_file: Optional[str],
    output_dir: str | Path,
) -> tuple[str, Optional[str]]:
    question_text = str(question or "").strip()
    answer_text = str(answer or "").strip()
    if not answer_text:
        return format_pdf_export_result("No answer available to export.", None)
    pdf_path = export_current_answer_to_pdf(
        question=question_text,
        answer=answer_text,
        source_file=str(source_file or "").strip() or None,
        output_dir=output_dir,
    )
    return format_pdf_export_result("PDF exported successfully.", pdf_path)


def export_chat_history_action(
    chat_history,
    selected_doc_label: Optional[str],
    last_source: Optional[str],
    output_dir: str | Path,
) -> tuple[str, Optional[str]]:
    messages = normalize_to_messages(chat_history)
    if not messages:
        return format_pdf_export_result("No chat history available to export.", None)
    selected_label = parse_doc_label(selected_doc_label)
    source_for_report = selected_label or (str(last_source or "").strip() or None)
    pdf_path = export_chat_history_to_pdf(
        messages=messages,
        selected_file=source_for_report,
        output_dir=output_dir,
    )
    return format_pdf_export_result("PDF exported successfully.", pdf_path)


def generate_pdf_report_action(
    *,
    report_request: str,
    selected_doc_label: Optional[str],
    service,
    output_dir: str | Path,
    report_type: str = "technical_paper",
    structured: bool = True,
    include_metadata: bool = False,
) -> tuple[str, Optional[str]] | tuple[str, Optional[str], dict]:
    req = str(report_request or "").strip()
    if not req:
        result = format_pdf_export_result_with_metadata(
            "Please enter a report request.",
            None,
            {"report_mode": "agent_structured" if structured else "legacy"},
        )
        return result if include_metadata else result[:2]

    service._reload_registry()
    selected_doc_id = parse_doc_id_from_label(selected_doc_label)
    doc: Optional[DocumentRecord] = None
    if selected_doc_id:
        doc = service._find_by_id(selected_doc_id)
        if doc is None:
            result = format_pdf_export_result_with_metadata(
                "Selected file is not available.",
                None,
            )
            return result if include_metadata else result[:2]
        if doc.status != "indexed":
            result = format_pdf_export_result_with_metadata(
                "Selected file is not indexed successfully. Please reprocess it first.",
                None,
            )
            return result if include_metadata else result[:2]
    inferred_report_type = (
        (report_type or "").strip().lower() if report_type else infer_report_type(req)
    )
    source_detection = detect_report_source_candidates(
        service=service,
        report_request=req,
        selected_doc_id=selected_doc_id,
    )
    inferred_report_type = (
        source_detection.get("inferred_report_type") or inferred_report_type
    )
    if source_detection["status"] == "no_source":
        result = format_pdf_export_result_with_metadata(
            source_detection["message"],
            None,
            {
                "report_mode": "agent_structured" if structured else "legacy",
                "source_selection_mode": "auto" if not selected_doc_id else "selected",
                "available_sources": source_detection["available_sources"],
                "candidate_sources": source_detection["candidate_sources"],
                "excluded_candidates": source_detection["excluded_candidates"],
                "inferred_report_type": inferred_report_type,
            },
        )
        return result if include_metadata else result[:2]
    if source_detection["status"] == "ambiguous":
        result = format_pdf_export_result_with_metadata(
            source_detection["message"],
            None,
            {
                "report_mode": "agent_structured" if structured else "legacy",
                "source_selection_mode": "auto",
                "available_sources": source_detection["available_sources"],
                "candidate_sources": source_detection["candidate_sources"],
                "excluded_candidates": source_detection["excluded_candidates"],
                "inferred_report_type": inferred_report_type,
            },
        )
        return result if include_metadata else result[:2]

    selected_records: list[DocumentRecord] = source_detection["selected_records"]
    if not selected_records:
        result = format_pdf_export_result_with_metadata(
            "No relevant indexed source was found for this report request.",
            None,
            {
                "report_mode": "agent_structured" if structured else "legacy",
                "source_selection_mode": "auto",
                "available_sources": source_detection["available_sources"],
                "candidate_sources": source_detection["candidate_sources"],
                "excluded_candidates": source_detection["excluded_candidates"],
                "inferred_report_type": inferred_report_type,
            },
        )
        return result if include_metadata else result[:2]

    try:
        selected_working_dirs = {
            rec.doc_id: (
                str(service._resolve_record_working_dir(rec))
                if hasattr(service, "_resolve_record_working_dir")
                else str(getattr(rec, "working_dir_rel", ""))
            )
            for rec in selected_records
        }

        def _query_func(question_text: str, rec: DocumentRecord) -> dict:
            # Every report section is restricted to the selected report document.
            # Wrong-source evidence is rejected before it reaches the PDF.
            if hasattr(service, "_query_document_with_source_guard"):
                result = service.run(
                    service._query_document_with_source_guard(
                        question=question_text,
                        rec=rec,
                        use_direct_vlm_on_query=False,
                    )
                )
            else:
                raw_answer = service.run(
                    service.query_existing_document(
                        question_text,
                        rec,
                        use_direct_vlm_on_query=False,
                    )
                )
                result = {
                    "ok": True,
                    "answer": str(raw_answer or ""),
                    "error": None,
                    "metadata": {
                        "selected_doc_id": rec.doc_id,
                        "selected_filename": rec.original_filename,
                    },
                }
            metadata = dict(result.get("metadata") or {})
            metadata.setdefault("selected_report_doc_id", rec.doc_id)
            metadata.setdefault("selected_report_filename", rec.original_filename)
            metadata.setdefault(
                "selected_report_working_dir", selected_working_dirs.get(rec.doc_id, "")
            )
            if metadata.get("selected_doc_id") not in {None, rec.doc_id}:
                return {
                    "ok": False,
                    "answer": "",
                    "error": "report section source mismatch",
                    "metadata": {
                        **metadata,
                        "source_validation": "failed",
                    },
                }
            return {
                "ok": bool(result.get("ok")),
                "answer": str(result.get("answer") or ""),
                "error": result.get("error"),
                "metadata": {
                    **metadata,
                    "source_validation": "passed" if result.get("ok") else "failed",
                },
            }

        if structured:
            structured_result = generate_structured_agent_report(
                request=req,
                doc_records=selected_records,
                output_dir=output_dir,
                query_func=_query_func,
                report_type=inferred_report_type,
            )
            result = format_pdf_export_result_with_metadata(
                "PDF report generated successfully.",
                structured_result["pdf_path"],
                {
                    "report_mode": "agent_structured",
                    "source_selection_mode": "auto"
                    if not selected_doc_id
                    else "selected",
                    "source_doc_id": structured_result["source_doc_id"],
                    "source_filename": structured_result["source_filename"],
                    "candidate_sources": source_detection["candidate_sources"],
                    "excluded_candidates": source_detection["excluded_candidates"],
                    "inferred_report_type": inferred_report_type,
                    "sections_generated": structured_result["sections_generated"],
                    "source_validation": structured_result["source_validation"],
                    "selected_report_doc_id": structured_result["source_doc_id"],
                    "selected_report_filename": structured_result["source_filename"],
                    "selected_report_working_dir": ", ".join(
                        selected_working_dirs.values()
                    ),
                },
            )
            return result if include_metadata else result[:2]

        pdf_path = generate_pdf_report_from_index(
            request=req,
            doc_record=selected_records[0],
            output_dir=output_dir,
            query_func=_query_func,
            report_type=inferred_report_type,
        )
        result = format_pdf_export_result_with_metadata(
            "PDF report generated successfully.",
            pdf_path,
            {
                "report_mode": "legacy_prompt_to_pdf",
                "source_selection_mode": "auto" if not selected_doc_id else "selected",
                "source_doc_id": selected_records[0].doc_id,
                "source_filename": selected_records[0].original_filename,
                "candidate_sources": source_detection["candidate_sources"],
                "excluded_candidates": source_detection["excluded_candidates"],
                "inferred_report_type": inferred_report_type,
                "source_validation": "passed",
            },
        )
        return result if include_metadata else result[:2]
    except Exception as exc:
        result = format_pdf_export_result_with_metadata(
            f"PDF report generation failed: {exc}",
            None,
        )
        return result if include_metadata else result[:2]
