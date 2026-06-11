from __future__ import annotations

from typing import Any

from backend.app.api._compat import APIRouter, File, JSONResponse, UploadFile
from backend.app.core.config import get_default_paths
from backend.app.core.runtime import get_backend_async_runtime
from backend.app.services.document_service import DocumentRegistryService
from backend.app.services.indexing_service import DocumentLifecycleService

router = APIRouter()


def _registry_payload() -> list[dict[str, Any]]:
    paths = get_default_paths()
    service = DocumentRegistryService(paths)
    return [service.serialize_document_record(rec) for rec in service.list_documents()]


@router.get("/documents")
def list_documents():
    return {"documents": _registry_payload()}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(None)):
    if file is None:
        return JSONResponse(
            content={"ok": False, "error": "Upload failed: missing file."},
            status_code=400,
        )
    try:
        content = await file.read()
        service = DocumentLifecycleService(get_default_paths())
        result = service.upload_document_bytes(
            filename=getattr(file, "filename", "") or "",
            content=content,
        )
        return result
    except Exception as exc:
        return JSONResponse(
            content={"ok": False, "error": f"Upload failed: {exc}"},
            status_code=500,
        )


@router.post("/documents/{doc_id}/index")
def index_document(doc_id: str):
    try:
        service = DocumentLifecycleService(get_default_paths())
        # Use the shared backend loop so index and chat reuse the same LightRAG
        # async lock context instead of creating a fresh request loop.
        result = get_backend_async_runtime().run(
            service.index_document_by_id(doc_id, force_reprocess=False)
        )
        if result.get("status_code") == 404:
            return JSONResponse(
                content={k: v for k, v in result.items() if k != "status_code"},
                status_code=404,
            )
        if result.get("ok"):
            return result
        return JSONResponse(content=result, status_code=500)
    except Exception as exc:
        return JSONResponse(
            content={"ok": False, "document": None, "error": f"Indexing failed: {exc}"},
            status_code=500,
        )


@router.post("/documents/{doc_id}/reprocess")
def reprocess_document(doc_id: str):
    try:
        service = DocumentLifecycleService(get_default_paths())
        # Reprocess follows the same single-loop rule as indexing to avoid
        # cross-loop lock reuse inside LightRAG shared storage.
        result = get_backend_async_runtime().run(
            service.index_document_by_id(doc_id, force_reprocess=True)
        )
        if result.get("status_code") == 404:
            return JSONResponse(
                content={k: v for k, v in result.items() if k != "status_code"},
                status_code=404,
            )
        if result.get("ok"):
            return result
        return JSONResponse(content=result, status_code=500)
    except Exception as exc:
        return JSONResponse(
            content={
                "ok": False,
                "document": None,
                "error": f"Reprocess failed: {exc}",
            },
            status_code=500,
        )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        service = DocumentRegistryService(get_default_paths())
        result = service.delete_document_result(doc_id)
        if result["ok"]:
            return result
        return JSONResponse(content=result, status_code=404)
    except Exception as exc:
        return JSONResponse(
            content={
                "ok": False,
                "deleted_doc_id": doc_id,
                "error": f"Unexpected delete error: {exc}",
            },
            status_code=500,
        )
