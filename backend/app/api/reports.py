from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from backend.app.api._compat import APIRouter, FileResponse, JSONResponse
from backend.app.core.config import get_default_paths
from backend.app.core.runtime import get_backend_async_runtime
from backend.app.services.indexing_service import DocumentLifecycleService
from backend.app.services.query_service import QueryServiceMixin
from backend.app.services.report_service import (
    export_chat_history_action,
    export_current_answer_action,
    generate_pdf_report_action,
)

router = APIRouter()


class ReportAPIService(DocumentLifecycleService, QueryServiceMixin):
    def run(self, coro):
        # Report generation reuses the same shared runtime as chat so query-only
        # calls do not reintroduce cross-loop LightRAG lock issues.
        return get_backend_async_runtime().run(coro)


def _error_response(message: str, *, status_code: int, details: dict | None = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details is not None:
        payload["details"] = details
    return JSONResponse(content=payload, status_code=status_code)


def _success_response(message: str, pdf_path: Path, reports_root: Path):
    filename = pdf_path.name
    # Expose only the report filename and a download route so the frontend does
    # not depend on local filesystem paths.
    return {
        "ok": True,
        "message": message,
        "filename": filename,
        "file_path": str(pdf_path),
        "open_url": f"/reports/download/{filename}?disposition=inline",
        "download_url": f"/reports/download/{filename}",
        "reports_root": str(reports_root),
    }


def _coerce_text(payload: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _coerce_selected_doc_label(
    service: ReportAPIService, payload: dict[str, Any]
) -> str | None:
    # React will usually send a selected doc id while Gradio-style helpers still
    # accept labels. Support both without forcing the frontend to know labels.
    selected_doc_label = _coerce_text(payload, "selected_doc_label")
    if selected_doc_label:
        return selected_doc_label
    selected_doc_id = _coerce_text(payload, "selected_doc_id", "selected_document_id")
    if not selected_doc_id:
        return None
    rec = service._find_by_id(selected_doc_id)
    if rec is None:
        return None
    return f"{rec.original_filename} [{rec.doc_id}]"


def _safe_report_path(filename: str, reports_root: Path) -> Path | None:
    raw = str(filename or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw:
        return None
    candidate = (reports_root / raw).resolve()
    try:
        candidate.relative_to(reports_root.resolve())
    except Exception:
        return None
    return candidate


@router.post("/reports/current-answer")
def report_current_answer(payload: dict | None = None):
    payload = payload or {}
    question = _coerce_text(payload, "question") or ""
    answer = _coerce_text(payload, "answer") or ""
    source_file = _coerce_text(payload, "source_file")
    reports_root = get_default_paths().reports_root.resolve()
    status, pdf_path_text = export_current_answer_action(
        question=question,
        answer=answer,
        source_file=source_file,
        output_dir=reports_root,
    )
    if not pdf_path_text:
        return _error_response(
            status,
            status_code=400,
            details={"source_file": source_file},
        )
    pdf_path = Path(pdf_path_text)
    return _success_response(status, pdf_path, reports_root)


@router.post("/reports/chat-history")
def report_chat_history(payload: dict | None = None):
    payload = payload or {}
    reports_root = get_default_paths().reports_root.resolve()
    service = ReportAPIService(get_default_paths())
    selected_doc_label = _coerce_selected_doc_label(service, payload)
    status, pdf_path_text = export_chat_history_action(
        chat_history=payload.get("messages") or payload.get("history") or [],
        selected_doc_label=selected_doc_label,
        last_source=_coerce_text(payload, "last_source", "source_file"),
        output_dir=reports_root,
    )
    if not pdf_path_text:
        return _error_response(
            status,
            status_code=400,
            details={"selected_doc_label": selected_doc_label},
        )
    pdf_path = Path(pdf_path_text)
    return _success_response(status, pdf_path, reports_root)


@router.post("/reports/agent")
def report_agent(payload: dict | None = None):
    payload = payload or {}
    reports_root = get_default_paths().reports_root.resolve()
    service = ReportAPIService(get_default_paths())
    selected_doc_label = _coerce_selected_doc_label(service, payload)
    if not selected_doc_label and _coerce_text(
        payload, "selected_doc_id", "selected_document_id"
    ):
        return _error_response(
            "Selected document was not found.",
            status_code=404,
        )
    # Let the service infer the report type when the frontend sends only a request.
    report_type = _coerce_text(payload, "report_type")
    structured = bool(payload.get("structured", True))
    status, pdf_path_text, metadata = generate_pdf_report_action(
        report_request=_coerce_text(payload, "report_request", "request") or "",
        selected_doc_label=selected_doc_label,
        service=service,
        output_dir=reports_root,
        report_type=report_type,
        structured=structured,
        include_metadata=True,
    )
    if not pdf_path_text:
        status_code = 400
        lowered = status.lower()
        if "not available" in lowered:
            status_code = 404
        elif "not indexed" in lowered:
            status_code = 409
        return _error_response(
            status,
            status_code=status_code,
            details={
                "selected_doc_label": selected_doc_label,
                **metadata,
            },
        )
    pdf_path = Path(pdf_path_text)
    response = _success_response(status, pdf_path, reports_root)
    response["metadata"] = metadata
    return response


@router.get("/reports/download/{filename}")
def report_download(filename: str, disposition: str = "attachment"):
    reports_root = get_default_paths().reports_root.resolve()
    pdf_path = _safe_report_path(filename, reports_root)
    # Reject traversal before touching the filesystem.
    if pdf_path is None:
        return _error_response(
            "Invalid report filename.",
            status_code=400,
        )
    if not pdf_path.exists() or not pdf_path.is_file():
        return _error_response(
            "Report file not found.",
            status_code=404,
        )
    header_disposition = "inline" if disposition == "inline" else "attachment"
    return FileResponse(
        path=str(pdf_path),
        filename=pdf_path.name,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{header_disposition}; filename="{pdf_path.name}"'
        },
    )
