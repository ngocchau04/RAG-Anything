from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pdf_exporter import export_structured_report_to_pdf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ReportSectionPlan:
    key: str
    title: str
    query: str


def infer_language(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "en"
    lowered = raw.lower()
    if any(token in lowered for token in ["tom tat", "bao cao", "tai lieu"]):
        return "vi"
    return "en"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _strip_source_reference_lines(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_refs = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("source:") or lowered.startswith("source file:"):
            continue
        if lowered in {
            "references",
            "references / source notes",
            "reference",
            "source notes",
        }:
            skip_refs = True
            continue
        if skip_refs:
            if (
                re.match(r"^[-*]\s*\[\d+\]\s+", line)
                or re.match(r"^\[\d+\]\s+", line)
                or line.startswith("- ")
            ):
                continue
            skip_refs = False
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines).strip()


def _dedupe_paragraphs(text: str) -> str:
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        normalized = _normalize_text(paragraph)
        if normalized in seen:
            continue
        if deduped:
            prev = _normalize_text(deduped[-1])
            if normalized in prev or prev in normalized:
                continue
        seen.add(normalized)
        deduped.append(paragraph)
    return "\n\n".join(deduped)


def plan_report_sections(
    report_type: str, request: str, language: str
) -> list[ReportSectionPlan]:
    report_type = (report_type or "technical_paper").strip().lower()
    if report_type == "summary":
        titles = [
            ("executive_summary", "Executive Summary"),
            ("key_findings", "1. Key Findings"),
            ("limitations", "2. Limitations"),
            ("conclusion", "3. Conclusion"),
        ]
    elif report_type == "table_figure_analysis":
        titles = [
            ("executive_summary", "Executive Summary"),
            ("tables_figures", "1. Tables / Figures Analysis"),
            ("results", "2. Results and Findings"),
            ("limitations", "3. Limitations"),
            ("conclusion", "4. Conclusion"),
        ]
    else:
        titles = [
            ("executive_summary", "Executive Summary"),
            ("introduction", "1. Introduction / Objective"),
            ("methodology", "2. Methodology"),
            ("dataset", "3. Dataset / Input Data"),
            ("formulas", "4. Key Formulas / Equations"),
            ("results", "5. Results and Findings"),
            ("tables_figures", "6. Tables / Figures Analysis"),
            ("discussion", "7. Discussion"),
            ("limitations", "8. Limitations"),
            ("conclusion", "9. Conclusion"),
        ]

    section_queries = {
        "executive_summary": f"Answer only from this file. Write a concise executive summary for this request: {request}",
        "introduction": "Answer only from this file. What is the introduction, objective, or problem addressed by this document?",
        "methodology": "Answer only from this file. Describe the methodology, model, or process used in the document.",
        "dataset": "Answer only from this file. Describe the dataset, input data, or experimental setup used in the document.",
        "formulas": "Answer only from this file. List the key formulas or equations relevant to the document.",
        "results": "Answer only from this file. Summarize the main results and findings in the document.",
        "tables_figures": "Answer only from this file. Summarize the important tables, figures, charts, and their implications.",
        "discussion": "Answer only from this file. Discuss the implications, interpretation, or analysis in the document.",
        "limitations": "Answer only from this file. List only the limitations, missing evidence, or uncertainties in the document.",
        "conclusion": "Answer only from this file. Provide a concise conclusion for the report request.",
        "key_findings": "Answer only from this file. Summarize the most important findings relevant to the report request.",
    }
    return [
        ReportSectionPlan(key=key, title=title, query=section_queries[key])
        for key, title in titles
    ]


def validate_section_sources(
    *,
    selected_doc_id: str,
    selected_filename: str,
    validation_metadata: dict[str, Any] | None,
) -> tuple[bool, str]:
    validation_metadata = validation_metadata or {}
    if validation_metadata.get("selected_doc_id") not in {None, selected_doc_id}:
        return False, "doc_id mismatch"
    if validation_metadata.get("selected_filename") not in {None, selected_filename}:
        return False, "filename mismatch"
    mismatched_sources = validation_metadata.get("mismatched_source_files") or []
    if mismatched_sources:
        return False, f"mismatched referenced files: {', '.join(mismatched_sources)}"
    return True, "passed"


def _insufficient_evidence_text(reason: str) -> str:
    return f"Insufficient evidence from the selected document. Validation detail: {reason}."


def build_report_markdown(
    *,
    title: str,
    source_file: str,
    source_doc_id: str,
    request: str,
    source_validation_status: str,
    inferred_report_type: str,
    sections: list[dict[str, Any]],
    references: list[str],
    agent_steps: list[str],
) -> str:
    lines = [f"# {title}", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append(section["content"] or "Insufficient evidence.")
        lines.append("")
    lines.extend(
        [
            "## References",
            *(f"- {reference}" for reference in references),
            "",
            "## Appendix: Agent Steps and Source Checks",
            f"- Source document name: {source_file}",
            f"- Source document id: {source_doc_id}",
            f"- Report request: {request}",
            f"- Inferred report type: {inferred_report_type}",
            f"- Source validation status: {source_validation_status}",
            *[f"- {step}" for step in agent_steps],
        ]
    )
    return "\n".join(lines).strip()


def _coerce_records(
    *, doc_record: Any | None = None, doc_records: list[Any] | None = None
) -> list[Any]:
    records = [record for record in (doc_records or []) if record is not None]
    if records:
        return records
    if doc_record is not None:
        return [doc_record]
    raise ValueError("At least one source document is required.")


def generate_structured_agent_report(
    *,
    request: str,
    doc_record: Any | None = None,
    doc_records: list[Any] | None = None,
    output_dir: str | Path,
    query_func: Callable[[str, Any], dict[str, Any]],
    report_type: str = "technical_paper",
) -> dict[str, Any]:
    req = (request or "").strip()
    if not req:
        raise ValueError("Please enter a report request.")

    records = _coerce_records(doc_record=doc_record, doc_records=doc_records)
    source_files = [
        str(getattr(record, "original_filename", "") or "N/A") for record in records
    ]
    source_doc_ids = [str(getattr(record, "doc_id", "") or "N/A") for record in records]
    source_file = ", ".join(source_files)
    source_doc_id = ", ".join(source_doc_ids)
    language = infer_language(req)
    title = "Structured Agent Report" if language != "vi" else "Structured Agent Report"
    plans = plan_report_sections(report_type, req, language)
    planned_outline = [plan.title for plan in plans]

    rendered_sections: list[dict[str, Any]] = []
    references: list[str] = []
    agent_steps: list[str] = [
        f"Planned outline: {', '.join(planned_outline)}",
        f"Report type: {report_type}",
        f"Selected sources: {source_file}",
    ]
    validation_passed = True

    for plan in plans:
        source_fragments: list[str] = []
        section_validation = "passed"
        for record in records:
            record_source_file = str(getattr(record, "original_filename", "") or "N/A")
            record_source_doc_id = str(getattr(record, "doc_id", "") or "N/A")
            query_result = query_func(plan.query, record)
            ok = bool(query_result.get("ok"))
            answer = _strip_source_reference_lines(
                str(query_result.get("answer") or "").strip()
            )
            is_valid, validation_reason = validate_section_sources(
                selected_doc_id=record_source_doc_id,
                selected_filename=record_source_file,
                validation_metadata=query_result.get("metadata"),
            )
            if not ok or not is_valid:
                validation_passed = False
                section_validation = "insufficient_evidence"
                source_fragments.append(
                    f"[{record_source_file}] "
                    f"{_insufficient_evidence_text(str(query_result.get('error') or validation_reason))}"
                )
            else:
                cleaned_answer = _dedupe_paragraphs(answer)
                if cleaned_answer:
                    source_fragments.append(f"[{record_source_file}] {cleaned_answer}")
                else:
                    validation_passed = False
                    section_validation = "insufficient_evidence"
                    source_fragments.append(
                        f"[{record_source_file}] "
                        f"{_insufficient_evidence_text('empty answer')}"
                    )
            references.append(
                f"{record_source_file} (doc_id={record_source_doc_id}) - {plan.title}: "
                f"{'passed' if ok and is_valid else 'insufficient_evidence'}"
            )
            agent_steps.append(
                f"Section '{plan.title}' sub-question for {record_source_file}: "
                f"{plan.query} | validation: {'passed' if ok and is_valid else 'insufficient_evidence'}"
            )

        section_content = "\n\n".join(source_fragments).strip()
        rendered_sections.append(
            {
                "key": plan.key,
                "title": plan.title,
                "content": section_content
                or _insufficient_evidence_text("empty answer"),
                "validation": section_validation,
            }
        )

    report_markdown = build_report_markdown(
        title=title,
        source_file=source_file,
        source_doc_id=source_doc_id,
        request=req,
        source_validation_status="passed" if validation_passed else "partial",
        inferred_report_type=report_type,
        sections=rendered_sections,
        references=references,
        agent_steps=agent_steps,
    )
    pdf_path = export_structured_report_to_pdf(
        report_title=title,
        source_file=source_file,
        source_doc_id=source_doc_id,
        user_request=req,
        report_body=report_markdown,
        output_dir=output_dir,
        filename_prefix="rag_agent_report",
        source_validation_status="passed" if validation_passed else "partial",
        sections_generated=[section["title"] for section in rendered_sections],
        report_mode="agent_structured",
        inferred_report_type=report_type,
    )
    return {
        "pdf_path": pdf_path,
        "sections_generated": [section["title"] for section in rendered_sections],
        "source_validation": "passed" if validation_passed else "partial",
        "agent_steps": agent_steps,
        "references": references,
        "report_markdown": report_markdown,
        "source_doc_id": source_doc_id,
        "source_filename": source_file,
    }


def generate_pdf_report_from_index(
    request: str,
    doc_record: Any,
    output_dir: str | Path,
    query_func: Callable[[str, Any], dict[str, Any]] | Callable[[str, Any], str],
    report_type: str = "technical_paper",
) -> Path:
    def _wrapped_query(question: str, rec: Any) -> dict[str, Any]:
        result = query_func(question, rec)
        if isinstance(result, dict):
            return result
        return {
            "ok": True,
            "answer": str(result or ""),
            "error": None,
            "metadata": {
                "selected_doc_id": getattr(rec, "doc_id", None),
                "selected_filename": getattr(rec, "original_filename", None),
            },
        }

    result = generate_structured_agent_report(
        request=request,
        doc_record=doc_record,
        output_dir=output_dir,
        query_func=_wrapped_query,
        report_type=report_type,
    )
    return result["pdf_path"]
