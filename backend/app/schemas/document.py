from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    doc_id: str
    original_filename: str
    stored_file_rel: str
    working_dir_rel: str
    file_type: str
    parser: str
    pdf_mode: str
    indexed_at: str
    status: str
    summary: str
    sha256: str
    source_metadata: dict[str, Any]
    visual_targets: list[dict[str, Any]]
    needs_reprocess: bool = False
    needs_reprocess_reason: str = ""
    error_message: str = ""


@dataclass
class UIState:
    processed: bool = False
    processing_summary: str = ""
    last_process_ts: float = 0.0
    ingest_count: int = 0
    query_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
