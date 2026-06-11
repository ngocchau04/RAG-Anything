from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    doc_id: str
    original_filename: str
    file_type: str
    parser: str
    pdf_mode: str
    indexed_at: str
    status: str
    summary: str
    sha256: str
    source_metadata: dict[str, Any]
    visual_targets: list[dict[str, Any]]
    stored_file_rel: str = ""
    working_dir_rel: str = ""
    needs_reprocess: bool = False
    needs_reprocess_reason: str = ""
    error_message: str = ""
    # Legacy WebUI/tests still instantiate records with absolute paths. Accept
    # those aliases and map them back to the new relative-path fields.
    stored_file_path: str = ""
    working_dir: str = ""

    def __post_init__(self) -> None:
        if not self.stored_file_rel and self.stored_file_path:
            self.stored_file_rel = self.stored_file_path
        if not self.working_dir_rel and self.working_dir:
            self.working_dir_rel = self.working_dir
        if not self.stored_file_path and self.stored_file_rel:
            self.stored_file_path = self.stored_file_rel
        if not self.working_dir and self.working_dir_rel:
            self.working_dir = self.working_dir_rel


@dataclass
class UIState:
    processed: bool = False
    processing_summary: str = ""
    last_process_ts: float = 0.0
    ingest_count: int = 0
    query_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
