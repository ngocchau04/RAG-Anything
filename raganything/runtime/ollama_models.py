from __future__ import annotations

from typing import Any, Optional

from backend.app.core.config import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_PROVIDER,
)


def normalize_ollama_host(host: str) -> str:
    # Keep host normalization in one place so backend and legacy example code
    # do not drift on trailing slashes or empty local-vs-Docker host values.
    text = str(host or "").strip()
    return text.rstrip("/") if text else "http://localhost:11434"


def resolve_ollama_embedding_model_name(
    requested_model: str,
    available_models: list[str],
) -> Optional[str]:
    # Normalize `model` vs `model:latest` so local Ollama installs do not fail
    # purely because the tag is omitted in config.
    cleaned_available = [
        str(name or "").strip() for name in available_models if str(name or "").strip()
    ]
    if not cleaned_available:
        return None

    lower_to_actual = {name.lower(): name for name in cleaned_available}

    def _candidate_match(candidate: str) -> Optional[str]:
        raw = str(candidate or "").strip()
        if not raw:
            return None
        lowered = raw.lower()
        if lowered in lower_to_actual:
            return lower_to_actual[lowered]
        if ":" not in raw:
            tagged = f"{raw}:latest".lower()
            if tagged in lower_to_actual:
                return lower_to_actual[tagged]
        if raw.endswith(":latest"):
            base = raw[: -len(":latest")].lower()
            if base in lower_to_actual:
                return lower_to_actual[base]
        return None

    priority = [requested_model]
    requested_text = str(requested_model or "").strip()
    if requested_text and ":" not in requested_text:
        priority.append(f"{requested_text}:latest")
    elif requested_text.endswith(":latest"):
        priority.append(requested_text[: -len(":latest")])

    # Local development prefers the already-installed nomic-embed-local family.
    priority.extend(
        [
            DEFAULT_LOCAL_EMBEDDING_MODEL,
            "nomic-embed-local",
            "nomic-embed-text",
            "nomic-embed-text:latest",
        ]
    )

    seen: set[str] = set()
    for candidate in priority:
        normalized = str(candidate or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved = _candidate_match(candidate)
        if resolved:
            return resolved
    return None


def build_ollama_embedding_diagnostic(
    *,
    embedding_provider: str,
    requested_model: str,
    embedding_dim: int,
    ollama_host: str,
    available_models: list[str],
    ollama_reachable: bool,
    connection_error: str | None = None,
) -> dict[str, Any]:
    normalized_host = normalize_ollama_host(ollama_host)
    diagnostic = {
        "embedding_provider": embedding_provider,
        "requested_model": str(requested_model or "").strip(),
        "resolved_model": None,
        "available_models": list(available_models),
        "embedding_dim": int(embedding_dim),
        "ollama_host": normalized_host,
        "ollama_reachable": bool(ollama_reachable),
        "suggestion": (
            "Run `ollama list` and set EMBEDDING_MODEL to an installed model such as "
            f"{DEFAULT_LOCAL_EMBEDDING_MODEL}."
        ),
    }
    if embedding_provider != DEFAULT_LOCAL_EMBEDDING_PROVIDER:
        diagnostic["suggestion"] = (
            "This fork requires local Ollama embeddings. Set "
            "EMBEDDING_PROVIDER=ollama and EMBEDDING_BINDING=ollama."
        )
        return diagnostic
    if connection_error:
        diagnostic["suggestion"] = (
            f"Start Ollama on {normalized_host} and verify the embedding model is installed. "
            f"Connection error: {connection_error}"
        )
        return diagnostic
    diagnostic["resolved_model"] = resolve_ollama_embedding_model_name(
        requested_model, available_models
    )
    return diagnostic
