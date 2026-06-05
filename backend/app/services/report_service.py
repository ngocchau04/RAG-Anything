from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from raganything.export import (
    export_chat_history_to_pdf,
    export_current_answer_to_pdf,
    generate_pdf_report_from_index,
)

from backend.app.schemas.document import DocumentRecord
from backend.app.services.query_service import (
    extract_candidate_filenames,
    normalize_to_messages,
)


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
) -> tuple[str, Optional[str]]:
    req = str(report_request or "").strip()
    if not req:
        return format_pdf_export_result("Please enter a report request.", None)

    service._reload_registry()
    all_docs = service.list_documents()

    selected_doc_id = parse_doc_id_from_label(selected_doc_label)
    doc: Optional[DocumentRecord] = None
    if selected_doc_id:
        doc = service._find_by_id(selected_doc_id)
        if doc is None:
            return format_pdf_export_result("Selected file is not available.", None)
        if doc.status != "indexed":
            return format_pdf_export_result(
                "Selected file is not indexed successfully. Please reprocess it first.",
                None,
            )
    indexed_docs = [r for r in all_docs if r.status == "indexed"]
    if not indexed_docs and doc is None:
        return format_pdf_export_result(
            "Please upload and process a file before generating a PDF report.",
            None,
        )
    else:
        mentioned_names = extract_candidate_filenames(req)
        if mentioned_names:
            for candidate in indexed_docs:
                if candidate.original_filename.lower() in mentioned_names:
                    doc = candidate
                    break
        if doc is None:
            if len(indexed_docs) > 1:
                return format_pdf_export_result(
                    "Please select a file before generating a PDF report.",
                    None,
                )
            doc = indexed_docs[0]

    assert doc is not None

    try:
        def _query_func(question_text: str, rec: DocumentRecord) -> str:
            return service.run(
                service.query_existing_document(
                    question_text,
                    rec,
                    use_direct_vlm_on_query=False,
                )
            )

        pdf_path = generate_pdf_report_from_index(
            request=req,
            doc_record=doc,
            output_dir=output_dir,
            query_func=_query_func,
        )
        return format_pdf_export_result("PDF report generated successfully.", pdf_path)
    except Exception as exc:
        return format_pdf_export_result(
            f"PDF report generation failed: {exc}",
            None,
        )
