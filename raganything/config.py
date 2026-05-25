"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

import os
from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value


@dataclass
class RAGAnythingConfig:
    """Configuration class for RAGAnything with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """Parser selection: 'mineru', 'docling', 'paddleocr', or 'simple_docx'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """Enable equation content processing."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: [
            x.strip()
            for x in get_env_value(
                "SUPPORTED_FILE_EXTENSIONS",
                ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
                str,
            ).split(",")
        ]
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: [
            x.strip()
            for x in get_env_value("CONTEXT_FILTER_CONTENT_TYPES", "text", str).split(
                ","
            )
        ]
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """Default content format for context extraction when processing documents."""

    # Path Handling Configuration
    # ---
    use_full_path: bool = field(default=get_env_value("USE_FULL_PATH", False, bool))
    """Whether to use full file path (True) or just basename (False) for file references in LightRAG."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value


@dataclass
class EmbeddingRuntimeConfig:
    provider: str
    model: str
    dim: int
    ollama_host: str
    provider_source: str
    model_source: str
    dim_source: str
    host_source: str


def resolve_embedding_runtime_config(
    env: dict | None = None, default_provider: str = "openai"
) -> EmbeddingRuntimeConfig:
    """Resolve embedding runtime config with provider-aware defaults."""
    values = env or os.environ

    provider_raw = values.get("EMBEDDING_PROVIDER")
    binding_raw = values.get("EMBEDDING_BINDING")
    provider = (provider_raw or binding_raw or default_provider).strip().lower()
    provider_source = (
        "env:EMBEDDING_PROVIDER"
        if provider_raw
        else ("env:EMBEDDING_BINDING" if binding_raw else f"default:{default_provider}")
    )

    model_raw = values.get("EMBEDDING_MODEL")
    dim_raw = values.get("EMBEDDING_DIM")
    host_raw = values.get("OLLAMA_HOST") or values.get("OLLAMA_BASE_URL")

    if provider == "ollama":
        model = (model_raw or "nomic-embed-local:latest").strip()
        if model == "text-embedding-3-large":
            raise RuntimeError(
                "Embedding provider/model mismatch: Ollama cannot use text-embedding-3-large. "
                "Set EMBEDDING_MODEL=nomic-embed-local:latest or another model from `ollama list`."
            )
        dim = int((dim_raw or "768").strip())
        host = (host_raw or "http://localhost:11434").strip()
        return EmbeddingRuntimeConfig(
            provider=provider,
            model=model,
            dim=dim,
            ollama_host=host,
            provider_source=provider_source,
            model_source=(
                "env:EMBEDDING_MODEL"
                if model_raw
                else "default:ollama(nomic-embed-local:latest)"
            ),
            dim_source=("env:EMBEDDING_DIM" if dim_raw else "default:ollama(768)"),
            host_source=(
                "env:OLLAMA_HOST/OLLAMA_BASE_URL"
                if host_raw
                else "default:ollama(http://localhost:11434)"
            ),
        )

    model = (model_raw or "text-embedding-3-large").strip()
    dim = int((dim_raw or "3072").strip())
    host = (host_raw or "http://localhost:11434").strip()
    return EmbeddingRuntimeConfig(
        provider=provider,
        model=model,
        dim=dim,
        ollama_host=host,
        provider_source=provider_source,
        model_source=(
            "env:EMBEDDING_MODEL" if model_raw else "default:openai(text-embedding-3-large)"
        ),
        dim_source=("env:EMBEDDING_DIM" if dim_raw else "default:openai(3072)"),
        host_source=(
            "env:OLLAMA_HOST/OLLAMA_BASE_URL"
            if host_raw
            else "default:ollama(http://localhost:11434)"
        ),
    )
