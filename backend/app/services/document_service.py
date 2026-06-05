from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from lightrag.utils import logger

from backend.app.core.config import SPECIAL_QUERY_MARKERS
from backend.app.schemas.document import DocumentRecord


def is_index_ready(working_dir: str) -> bool:
    wd = Path(working_dir)
    required = [wd / "vdb_chunks.json", wd / "kv_store_text_chunks.json"]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


class RegistryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_load_migrated = False

    @staticmethod
    def _path_to_rel(path_value: Optional[str], expected_prefix: str) -> Optional[str]:
        if not path_value:
            return None
        raw = str(path_value).replace("\\", "/").strip()
        if not raw:
            return None
        if raw.startswith(f"{expected_prefix}/") or raw == expected_prefix:
            return raw
        marker = f"/{expected_prefix}/"
        if marker in raw:
            tail = raw.split(marker, 1)[1]
            return f"{expected_prefix}/{tail}"
        marker2 = f"{expected_prefix}/"
        idx = raw.find(marker2)
        if idx >= 0:
            return raw[idx:]
        return None

    def _from_raw_record(self, item: dict[str, Any]) -> tuple[DocumentRecord, bool]:
        migrated = False
        stored_rel = item.get("stored_file_rel")
        working_rel = item.get("working_dir_rel")
        if not stored_rel:
            stored_rel = self._path_to_rel(
                item.get("stored_file_path"), "webui_uploads"
            )
            migrated = migrated or bool(stored_rel)
        if not working_rel:
            working_rel = self._path_to_rel(item.get("working_dir"), "webui_docs")
            migrated = migrated or bool(working_rel)

        needs_reprocess = bool(item.get("needs_reprocess", False))
        reason = str(item.get("needs_reprocess_reason", "") or "").strip()
        if not stored_rel or not working_rel:
            needs_reprocess = True
            if not reason:
                reason = (
                    "could not migrate legacy absolute path(s) into "
                    "webui_uploads/ or webui_docs/"
                )

        rec = DocumentRecord(
            doc_id=str(item.get("doc_id", "")),
            original_filename=str(item.get("original_filename", "")),
            stored_file_rel=str(stored_rel or ""),
            working_dir_rel=str(working_rel or ""),
            file_type=str(item.get("file_type", "")),
            parser=str(item.get("parser", "")),
            pdf_mode=str(item.get("pdf_mode", "auto")),
            indexed_at=str(item.get("indexed_at", "")),
            status=str(item.get("status", "indexed")),
            summary=str(item.get("summary", "")),
            sha256=str(item.get("sha256", "")),
            source_metadata=dict(item.get("source_metadata", {}) or {}),
            visual_targets=list(item.get("visual_targets", []) or []),
            needs_reprocess=needs_reprocess,
            needs_reprocess_reason=reason,
            error_message=str(item.get("error_message", "") or ""),
        )
        migrated = migrated or ("stored_file_path" in item) or ("working_dir" in item)
        return rec, migrated

    def load(self) -> list[DocumentRecord]:
        self._last_load_migrated = False
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            records: list[DocumentRecord] = []
            migrated = False
            for item in raw:
                if not isinstance(item, dict):
                    continue
                rec, rec_migrated = self._from_raw_record(item)
                if not rec.doc_id:
                    continue
                records.append(rec)
                migrated = migrated or rec_migrated
            self._last_load_migrated = migrated
            return records
        except Exception:
            logger.warning("Registry file is invalid. Starting with empty registry.")
            return []

    def save(self, records: list[DocumentRecord]) -> None:
        payload = [asdict(r) for r in records]
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class DocumentServiceMixin:
    @staticmethod
    def _sanitize_error_text(value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        text = re.sub(r"[A-Za-z]:\\\\[^\\s]+", "[host-path]", text)
        text = re.sub(r"/app/[^\s]+", "[container-path]", text)
        return text

    @staticmethod
    def serialize_document_record(rec: DocumentRecord) -> dict[str, Any]:
        created_at = str(rec.indexed_at or "").strip() or None
        return {
            "doc_id": rec.doc_id,
            "filename": rec.original_filename,
            "status": rec.status,
            "file_type": rec.file_type,
            "stored_file_rel": rec.stored_file_rel,
            "working_dir_rel": rec.working_dir_rel,
            "created_at": created_at,
            "updated_at": created_at,
            "error_message": DocumentServiceMixin._sanitize_error_text(
                rec.error_message
            ),
            "needs_reprocess": bool(rec.needs_reprocess),
        }

    def serialize_document_result(self, rec: DocumentRecord) -> dict[str, Any]:
        payload = self.serialize_document_record(rec)
        payload["stored_file_rel"] = rec.stored_file_rel
        payload["working_dir_rel"] = rec.working_dir_rel
        return payload

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.registry)

    def _reload_registry(self) -> None:
        if self.registry_store.path.exists():
            self.registry = self.registry_store.load()

    def render_registry_text(self) -> str:
        if not self.registry:
            return "No indexed files yet."
        lines = ["Indexed files:"]
        for rec in self.registry:
            lines.append(f"- {rec.original_filename} [{rec.doc_id}] ({rec.status})")
        return "\n".join(lines)

    def _find_by_hash(self, sha256: str) -> Optional[DocumentRecord]:
        return next((r for r in self.registry if r.sha256 == sha256), None)

    def _find_by_id(self, doc_id: str) -> Optional[DocumentRecord]:
        return next((r for r in self.registry if r.doc_id == doc_id), None)

    @staticmethod
    def _as_rel(path: Path, root: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    def _resolve_rel(self, rel_path: str) -> Path:
        return (self.paths.storage_root / Path(rel_path)).resolve()

    def _resolve_record_stored_file(self, rec: DocumentRecord) -> Path:
        return self._resolve_rel(rec.stored_file_rel)

    def _resolve_record_working_dir(self, rec: DocumentRecord) -> Path:
        return self._resolve_rel(rec.working_dir_rel)

    def _record_index_status(self, rec: DocumentRecord) -> tuple[bool, str]:
        if rec.status != "indexed":
            detail = rec.error_message or rec.needs_reprocess_reason or rec.status
            return False, f"not indexed: {detail}"
        if rec.needs_reprocess:
            detail = rec.needs_reprocess_reason or "unknown reason"
            return False, f"registry path incompatible: {detail}"
        if not rec.working_dir_rel:
            return False, "registry path incompatible: missing working_dir_rel"
        working_dir = self._resolve_record_working_dir(rec)
        if not is_index_ready(str(working_dir)):
            return False, f"index path missing: {working_dir}"
        return True, ""

    def _load_text_chunks(self, rec: DocumentRecord) -> list[str]:
        wd = self._resolve_record_working_dir(rec)
        path = wd / "kv_store_text_chunks.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        texts: list[str] = []
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                    elif isinstance(value, str):
                        if (
                            "[PDF Table" in value
                            or "[PDF Equation" in value
                            or "[PDF Visual Description" in value
                            or str(key).lower() in {"content", "text"}
                        ):
                            texts.append(value)
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, str):
                texts.append(item)
        return texts

    def _find_marker_chunks(self, rec: DocumentRecord, marker_kind: str) -> list[str]:
        marker = SPECIAL_QUERY_MARKERS.get(marker_kind)
        if not marker:
            return []
        return [c for c in self._load_text_chunks(rec) if marker in c]

    def delete_document_result(self, doc_id: str) -> dict[str, Any]:
        rec = self._find_by_id(doc_id)
        if rec is None:
            return {
                "ok": False,
                "deleted_doc_id": doc_id,
                "message": "Document not found.",
                "error": "Document not found.",
            }
        self.rag_cache.pop(doc_id, None)
        wd = self._resolve_record_working_dir(rec)
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        stored_file = self._resolve_record_stored_file(rec)
        shared_upload = False
        for other in self.registry:
            if other.doc_id == rec.doc_id:
                continue
            if other.stored_file_rel == rec.stored_file_rel:
                shared_upload = True
                break
        if (not shared_upload) and stored_file.exists():
            try:
                stored_file.unlink()
            except Exception:
                pass
        self.registry = [r for r in self.registry if r.doc_id != doc_id]
        self.registry_store.save(self.registry)
        return {
            "ok": True,
            "deleted_doc_id": rec.doc_id,
            "message": "Document deleted successfully.",
        }

    def delete_document(self, doc_id: str) -> str:
        rec = self._find_by_id(doc_id)
        result = self.delete_document_result(doc_id)
        if result["ok"]:
            return f"Deleted indexed file: {rec.original_filename if rec else doc_id}"
        return "Please select an indexed file first."


class DocumentRegistryService(DocumentServiceMixin):
    def __init__(self, paths):
        self.paths = paths
        self.registry_store = RegistryStore(paths.registry_path)
        self.registry: list[DocumentRecord] = self.registry_store.load()
        if self.registry_store._last_load_migrated:
            self.registry_store.save(self.registry)
        self.rag_cache: dict[str, Any] = {}

    def save_registry(self) -> None:
        self.registry_store.save(self.registry)

    def upsert_record(self, record: DocumentRecord) -> None:
        self.registry = [r for r in self.registry if r.doc_id != record.doc_id]
        self.registry.append(record)
        self.save_registry()

    def create_uploaded_record(
        self,
        *,
        doc_id: str,
        filename: str,
        stored_file_rel: str,
        working_dir_rel: str,
        file_type: str,
        sha256: str,
        status: str = "uploaded",
        parser: str = "",
        pdf_mode: str = "auto",
        error_message: str = "",
        needs_reprocess: bool = False,
        needs_reprocess_reason: str = "",
        source_metadata: Optional[dict[str, Any]] = None,
        visual_targets: Optional[list[dict[str, Any]]] = None,
    ) -> DocumentRecord:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        return DocumentRecord(
            doc_id=doc_id,
            original_filename=filename,
            stored_file_rel=stored_file_rel,
            working_dir_rel=working_dir_rel,
            file_type=file_type,
            parser=parser,
            pdf_mode=pdf_mode,
            indexed_at=timestamp,
            status=status,
            summary="",
            sha256=sha256,
            source_metadata=source_metadata or {},
            visual_targets=visual_targets or [],
            needs_reprocess=needs_reprocess,
            needs_reprocess_reason=needs_reprocess_reason,
            error_message=error_message,
        )
