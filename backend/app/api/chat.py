from __future__ import annotations
from typing import Any, Optional

from backend.app.api._compat import APIRouter, JSONResponse
from backend.app.core.config import get_default_paths
from backend.app.core.runtime import get_backend_async_runtime
from backend.app.services.indexing_service import (
    DocumentLifecycleService,
    OllamaEmbeddingRuntimeError,
)
from backend.app.services.query_service import QueryServiceMixin, normalize_to_messages
from lightrag.utils import logger

router = APIRouter()


class ChatService(DocumentLifecycleService, QueryServiceMixin):
    """Bind the existing document/index runtime to the query mixin for API use."""


def _error_response(message: str, *, status_code: int, detail: Optional[dict] = None):
    # Keep API errors aligned with the existing document endpoints: a short
    # `error` string plus optional structured context when useful to the UI.
    payload: dict[str, Any] = {"ok": False, "error": message}
    if detail is not None:
        payload["details"] = detail
    return JSONResponse(content=payload, status_code=status_code)


def _coerce_payload_value(payload: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _chat_impl(payload: dict | None = None):
    # Accept a minimal JSON payload so the current frontend can evolve without
    # introducing Pydantic schemas in this migration step.
    payload = payload or {}
    question = _coerce_payload_value(payload, "message", "query", "question")
    # Accept both the current frontend field name and a more explicit alias so
    # API clients can evolve without breaking chat selection.
    selected_doc_id = _coerce_payload_value(
        payload, "selected_doc_id", "selected_document_id"
    )
    mode = str(payload.get("mode", "") or "").strip().lower() or "selected_document"
    require_selected_document = bool(payload.get("require_selected_document", False))
    history = normalize_to_messages(payload.get("history"))

    if not question:
        return _error_response(
            "Chat failed: message cannot be empty.",
            status_code=400,
        )

    if require_selected_document and not selected_doc_id:
        return _error_response(
            "Chat failed: please select an indexed document first.",
            status_code=400,
        )

    service = ChatService(get_default_paths())
    selected_doc_filename: Optional[str] = None
    selected_working_dir: Optional[str] = None
    selected_doc_status: Optional[str] = None

    if selected_doc_id:
        # When the React UI sends an explicit selection, fail early with a clear
        # state error instead of letting generic routing messages leak through.
        rec = service._find_by_id(selected_doc_id)
        if rec is None:
            return _error_response(
                "Chat failed: selected document was not found.",
                status_code=404,
            )
        ok, reason = service._record_index_status(rec)
        if not ok:
            return _error_response(
                "Chat failed: selected document is not indexed or its index is unavailable.",
                status_code=409,
                detail={
                    "doc_id": rec.doc_id,
                    "filename": rec.original_filename,
                    "reason": reason,
                },
            )
        selected_doc_filename = rec.original_filename
        selected_working_dir = str(service._resolve_record_working_dir(rec))
        selected_doc_status = rec.status

    use_corpus_mode = mode == "corpus" or (
        not require_selected_document and not selected_doc_id
    )
    try:
        # Selected-doc chat remains available as an optional filter, but corpus
        # mode is now the default multi-file workflow when no selection is sent.
        logger.info(
            "Chat request start: mode=%s doc_id=%s filename=%s status=%s working_dir=%s runtime_loop=%s history_count=%s",
            "corpus" if use_corpus_mode else "selected_document",
            selected_doc_id,
            selected_doc_filename,
            selected_doc_status,
            selected_working_dir,
            get_backend_async_runtime().loop_id,
            len(history),
        )
        if use_corpus_mode:
            result = get_backend_async_runtime().run(
                service.query_corpus_with_metadata(
                    question,
                    selected_doc_id=selected_doc_id,
                    history=history,
                    use_direct_vlm_on_query=False,
                )
            )
        else:
            result = get_backend_async_runtime().run(
                service.query_with_metadata(
                    question,
                    selected_doc_id=selected_doc_id,
                    history=history,
                    use_direct_vlm_on_query=False,
                )
            )
    except OllamaEmbeddingRuntimeError as exc:
        logger.warning(
            "Chat request failed with Ollama runtime error: doc_id=%s filename=%s runtime_loop=%s error=%s",
            selected_doc_id,
            selected_doc_filename,
            get_backend_async_runtime().loop_id,
            exc,
        )
        return _error_response(
            str(exc),
            status_code=503,
            detail=exc.details,
        )
    except Exception as exc:
        logger.exception(
            "Chat request failed: doc_id=%s filename=%s working_dir=%s runtime_loop=%s",
            selected_doc_id,
            selected_doc_filename,
            selected_working_dir,
            get_backend_async_runtime().loop_id,
        )
        return _error_response(
            f"Chat failed: {exc}",
            status_code=500,
            detail={
                "selected_doc_id": selected_doc_id,
                "selected_document_filename": selected_doc_filename,
                "selected_document_status": selected_doc_status,
                "working_dir": selected_working_dir,
                "runtime_loop_id": get_backend_async_runtime().loop_id,
            },
        )

    if not result.get("ok"):
        error_message = str(result.get("error") or "Chat failed.")
        logger.warning(
            "Chat request returned service error: doc_id=%s filename=%s working_dir=%s runtime_loop=%s error=%s",
            selected_doc_id,
            selected_doc_filename,
            selected_working_dir,
            get_backend_async_runtime().loop_id,
            error_message,
        )
        lowered = error_message.lower()
        if "selected file is not available" in lowered:
            return _error_response(
                "Chat failed: selected document was not found.",
                status_code=404,
            )
        if "please upload and process files first" in lowered:
            return _error_response(error_message, status_code=409)
        if "no relevant indexed file found" in lowered:
            return _error_response(
                error_message,
                status_code=404,
                detail=result.get("metadata"),
            )
        if "i could not find relevant information in the uploaded files" in lowered:
            return _error_response(
                error_message,
                status_code=404,
                detail=result.get("metadata"),
            )
        if "please enter a question" in lowered:
            return _error_response(error_message, status_code=400)
        if "multiple indexed files" in lowered:
            return _error_response(error_message, status_code=400)
        if "source does not match" in lowered:
            return _error_response(
                error_message,
                status_code=409,
                detail=result.get("metadata"),
            )
        return _error_response(error_message, status_code=400)
    if str(result.get("answer") or "").strip().lower().startswith("query failed:"):
        # Treat service-layer query failures as API failures so the frontend
        # does not render a backend exception as a successful assistant answer.
        logger.warning(
            "Chat request returned fake-success query failure: doc_id=%s filename=%s working_dir=%s runtime_loop=%s error=%s",
            selected_doc_id,
            selected_doc_filename,
            selected_working_dir,
            get_backend_async_runtime().loop_id,
            str(result.get("answer")).strip(),
        )
        return _error_response(
            str(result.get("answer")).strip(),
            status_code=500,
            detail={
                "selected_doc_id": selected_doc_id,
                "selected_document_filename": selected_doc_filename,
                "working_dir": selected_working_dir,
                "runtime_loop_id": get_backend_async_runtime().loop_id,
            },
        )

    result_document = result.get("document") or {}
    result_metadata = dict(result.get("metadata") or {})
    if selected_doc_id:
        result_metadata.setdefault("selected_doc_id", selected_doc_id)
    if selected_doc_filename:
        result_metadata.setdefault("selected_filename", selected_doc_filename)
    if selected_working_dir:
        result_metadata.setdefault("working_dir", selected_working_dir)
    if selected_doc_id and result_document.get("doc_id") not in {None, selected_doc_id}:
        return _error_response(
            "Chat response source does not match the currently selected document.",
            status_code=409,
            detail={
                "selected_doc_id": selected_doc_id,
                "selected_filename": selected_doc_filename,
                "selected_status": selected_doc_status,
                "working_dir": selected_working_dir,
                "response_document": result_document,
                "response_metadata": result_metadata,
            },
        )
    if selected_doc_id and result_metadata.get("selected_doc_id") not in {
        None,
        selected_doc_id,
    }:
        return _error_response(
            "Chat response source does not match the currently selected document.",
            status_code=409,
            detail={
                "selected_doc_id": selected_doc_id,
                "selected_filename": selected_doc_filename,
                "selected_status": selected_doc_status,
                "working_dir": selected_working_dir,
                "response_document": result_document,
                "response_metadata": result_metadata,
            },
        )
    if (
        selected_working_dir
        and result_metadata.get("working_dir")
        and result_metadata.get("working_dir") != selected_working_dir
    ):
        return _error_response(
            "Chat response source does not match the currently selected document.",
            status_code=409,
            detail={
                "selected_doc_id": selected_doc_id,
                "selected_filename": selected_doc_filename,
                "selected_status": selected_doc_status,
                "working_dir": selected_working_dir,
                "response_document": result_document,
                "response_metadata": result_metadata,
            },
        )

    logger.info(
        "Chat request succeeded: doc_id=%s filename=%s working_dir=%s runtime_loop=%s",
        selected_doc_id,
        selected_doc_filename or (result.get("document") or {}).get("filename"),
        selected_working_dir,
        get_backend_async_runtime().loop_id,
    )

    return {
        "ok": True,
        "answer": result.get("answer", ""),
        "document": result_document,
        "sources": result.get("sources", []),
        "metadata": result_metadata,
        "history": history,
    }


@router.post("/chat")
def chat(payload: dict | None = None):
    return _chat_impl(payload)


@router.post("/chat/corpus")
def chat_corpus(payload: dict | None = None):
    payload = dict(payload or {})
    payload["mode"] = "corpus"
    payload["require_selected_document"] = False
    return _chat_impl(payload)
