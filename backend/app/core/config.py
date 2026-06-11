from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCAL_EMBEDDING_PROVIDER = "ollama"
DEFAULT_LOCAL_EMBEDDING_BINDING = "ollama"
DEFAULT_LOCAL_EMBEDDING_MODEL = "nomic-embed-local:latest"
DEFAULT_LOCAL_EMBEDDING_DIM = 768
DEFAULT_LOCAL_OLLAMA_HOST = "http://localhost:11434"


@dataclass(frozen=True)
class AppPaths:
    registry_path: Path
    uploads_dir: Path
    docs_root: Path
    reports_root: Path

    @property
    def storage_root(self) -> Path:
        return self.registry_path.parent.resolve()


def get_default_paths() -> AppPaths:
    return AppPaths(
        registry_path=Path(
            os.getenv("WEBUI_REGISTRY_PATH", "./rag_storage/webui_registry.json")
        ).resolve(),
        uploads_dir=Path(
            os.getenv("WEBUI_UPLOADS_DIR", "./rag_storage/webui_uploads")
        ).resolve(),
        docs_root=Path(
            os.getenv("WEBUI_DOCS_ROOT", "./rag_storage/webui_docs")
        ).resolve(),
        reports_root=Path(
            os.getenv("WEBUI_REPORTS_ROOT", "./output/reports")
        ).resolve(),
    )


SUPPORTED_QUERY_LOAD_PARSERS = {"mineru", "docling", "paddleocr", "simple_docx"}
SPECIAL_QUERY_MARKERS = {
    "table": "[PDF Table",
    "equation": "[PDF Equation",
    "figure": "[PDF Visual Description",
}
