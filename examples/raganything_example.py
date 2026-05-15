#!/usr/bin/env python
"""
Example script demonstrating parser integration with RAGAnything
"""

import os
import json
import argparse
import asyncio
import base64
import logging
import logging.config
from functools import partial
from pathlib import Path
from typing import Optional, List
from time import perf_counter
import random
import numpy as np

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def _is_windows_winerror5(exc: Exception) -> bool:
    msg = str(exc)
    return "WinError 5" in msg or "PermissionError: [WinError 5]" in msg


def _is_gemini_quota_exceeded(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
        or "rate limit" in msg.lower()
    )


def _is_index_ready(working_dir: str) -> bool:
    """Check whether LightRAG index artifacts were generated."""
    wd = Path(working_dir)
    expected = [
        wd / "vdb_chunks.json",
        wd / "kv_store_text_chunks.json",
    ]
    return any(p.exists() and p.stat().st_size > 0 for p in expected)


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }


def _has_vision_provider(api_key: Optional[str], base_url: Optional[str]) -> bool:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return True
    if os.getenv("LLM_BINDING_API_KEY"):
        return True
    if api_key and (base_url or os.getenv("LLM_BINDING_HOST")):
        return True
    return False


def _detect_provider_name(base_url: Optional[str]) -> str:
    host = (base_url or os.getenv("LLM_BINDING_HOST") or "").lower()
    if "generativelanguage" in host or "googleapis" in host:
        return "gemini-openai-compatible"
    if "openai" in host:
        return "openai-compatible"
    if host:
        return "openai-compatible-custom"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini-openai-compatible"
    return "unknown"


def _is_pdf_file(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _parse_page_range(page_range: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not page_range:
        return None, None
    raw = page_range.strip()
    if "-" not in raw:
        raise ValueError("page-range must be in format start-end, e.g. 1-3")
    start_s, end_s = raw.split("-", 1)
    start = int(start_s.strip())
    end = int(end_s.strip())
    if start <= 0 or end <= 0 or end < start:
        raise ValueError("page-range must satisfy 1 <= start <= end")
    return start, end


def _parse_page_selection(
    page_range: Optional[str], max_pages: Optional[int]
) -> Optional[List[int]]:
    """Parse page selection syntax like '3,5,7-13' into 0-based page indexes.

    Returns None when no explicit page selection is provided.
    """
    if not page_range:
        return None
    indexes: set[int] = set()
    for token in [t.strip() for t in page_range.split(",") if t.strip()]:
        if "-" in token:
            s, e = token.split("-", 1)
            start = int(s)
            end = int(e)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(
                    "page-range token must satisfy 1 <= start <= end, e.g. 3-5"
                )
            for p in range(start, end + 1):
                indexes.add(p - 1)
        else:
            p = int(token)
            if p <= 0:
                raise ValueError("page-range page number must be >= 1")
            indexes.add(p - 1)
    ordered = sorted(indexes)
    if max_pages is not None and max_pages > 0:
        ordered = ordered[:max_pages]
    return ordered


def _build_vision_target_patterns(vision_target: str) -> List[str]:
    import re

    target = (vision_target or "").strip()
    if not target:
        return []

    nums = re.findall(r"\d+", target)
    if not nums:
        return [re.escape(target)]

    n = nums[0]
    return [
        rf"\bfig\.?\s*{n}\b",
        rf"\bfigure\s*{n}\b",
        rf"\bfigure\s*{n}\s*:",
    ]


def _resolve_vision_target_page_from_text_blocks(
    text_blocks: List[dict], vision_target: Optional[str]
) -> Optional[int]:
    import re

    if not vision_target:
        return None
    patterns = _build_vision_target_patterns(vision_target)
    if not patterns:
        return None

    for block in text_blocks:
        text = str(block.get("text", "") or "")
        page_idx = block.get("page_idx")
        if page_idx is None:
            continue
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return int(page_idx)
    return None


def _extract_text_from_pdf_fast(
    pdf_path: str,
    max_pages: Optional[int] = None,
    page_range: Optional[str] = None,
) -> List[dict]:
    selected_pages = _parse_page_selection(page_range, max_pages)
    page_counter = 0
    blocks: List[dict] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        try:
            total = len(doc)
            page_indices = (
                selected_pages if selected_pages is not None else range(total)
            )
            for i in page_indices:
                if i < 0 or i >= total:
                    continue
                if max_pages is not None and page_counter >= max_pages:
                    break
                text = (doc[i].get_text("text") or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text, "page_idx": i})
                page_counter += 1
        finally:
            doc.close()
        return blocks
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            page_indices = (
                selected_pages if selected_pages is not None else range(total)
            )
            for i in page_indices:
                if i < 0 or i >= total:
                    continue
                if max_pages is not None and page_counter >= max_pages:
                    break
                text = (pdf.pages[i].extract_text() or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text, "page_idx": i})
                page_counter += 1
        return blocks
    except Exception as exc:
        raise RuntimeError(
            "PDF fast mode requires PyMuPDF (fitz) or pdfplumber. Install one of:\n"
            "1) .\\.venv\\Scripts\\python.exe -m pip install pymupdf\n"
            "2) .\\.venv\\Scripts\\python.exe -m pip install pdfplumber"
        ) from exc


def _extract_tables_from_pdf_pdfplumber(
    pdf_path: str,
    max_pages: Optional[int] = None,
    page_range: Optional[str] = None,
) -> List[dict]:
    try:
        import pdfplumber
    except Exception:
        logger.warning(
            "pdfplumber not installed; table extraction skipped. Install with: "
            "python -m pip install pdfplumber"
        )
        return []

    selected_pages = _parse_page_selection(page_range, max_pages)
    blocks: List[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            page_indices = (
                selected_pages if selected_pages is not None else range(total)
            )
            scanned = 0
            for i in page_indices:
                if i < 0 or i >= total:
                    continue
                if max_pages is not None and max_pages > 0 and scanned >= max_pages:
                    break
                scanned += 1
                tables = pdf.pages[i].extract_tables() or []
                for t_idx, rows in enumerate(tables, start=1):
                    if not rows:
                        continue
                    width = max(len(r or []) for r in rows)
                    normalized = []
                    for r in rows:
                        rr = [(c or "").strip().replace("\n", " ") for c in (r or [])]
                        if len(rr) < width:
                            rr.extend([""] * (width - len(rr)))
                        normalized.append(rr)
                    if not normalized:
                        continue
                    header = normalized[0]
                    md_lines = [
                        f"[PDF Table | page={i + 1} | table={t_idx}]",
                        f"| {' | '.join(header)} |",
                        f"| {' | '.join(['---'] * width)} |",
                    ]
                    for body in normalized[1:]:
                        md_lines.append(f"| {' | '.join(body)} |")
                    blocks.append(
                        {
                            "type": "text",
                            "text": "\n".join(md_lines),
                            "page_idx": i,
                            "source": "pdfplumber",
                        }
                    )
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", str(exc))
        return []

    if not blocks:
        logger.info("No extractable tables found by pdfplumber on selected pages.")
    return blocks


def _render_pdf_pages_for_vision(
    pdf_path: str,
    tmp_dir: str,
    max_pages: Optional[int] = None,
    page_range: Optional[str] = None,
) -> List[dict]:
    import fitz

    selected_pages = _parse_page_selection(page_range, max_pages)
    out_dir = Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_items: List[dict] = []
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        page_indices = selected_pages if selected_pages is not None else range(total)
        rendered = 0
        for i in page_indices:
            if i < 0 or i >= total:
                continue
            if max_pages is not None and max_pages > 0 and rendered >= max_pages:
                break
            pix = doc[i].get_pixmap(dpi=144, alpha=False)
            img_path = out_dir / f"page_{i + 1}.png"
            pix.save(str(img_path))
            image_items.append(
                {
                    "type": "image",
                    "img_path": str(img_path),
                    "image_caption": [f"Rendered PDF page {i + 1}"],
                    "image_footnote": [],
                    "page_idx": i,
                    "source": "pdf_page_render",
                }
            )
            rendered += 1
    finally:
        doc.close()
    return image_items


async def _ensure_ollama_model_available(host: str, model: str) -> None:
    import aiohttp

    url = f"{host.rstrip('/')}/api/tags"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Ollama host check failed ({resp.status}). "
                    f"URL={url} body={body[:200]}"
                )
            data = await resp.json()
    models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
    aliases = set()
    for m in models:
        aliases.add(m)
        aliases.add(m.split(":")[0])
    if model not in aliases:
        raise RuntimeError(
            f"Ollama model '{model}' not found on host {host}. "
            f"Run: ollama pull {model}"
        )


def _is_vision_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = [
        "does not support image",
        "image input is not supported",
        "invalid image",
        "vision is not supported",
        "unsupported content type",
    ]
    return any(k in msg for k in keywords)


def _read_image_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


async def _describe_image_with_vision(
    image_path: str,
    prompt: str,
    api_key: str,
    base_url: Optional[str],
    vision_model: str,
    fallback_vision_model: Optional[str] = None,
) -> str:
    image_data = _read_image_base64(image_path)
    models_to_try = [vision_model]
    if fallback_vision_model and fallback_vision_model != vision_model:
        models_to_try.append(fallback_vision_model)

    last_exc = None
    for idx, model_name in enumerate(models_to_try):
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=120.0,
            )
            async with client:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                    ],
                )
            content = resp.choices[0].message.content if resp.choices else None
            if not content:
                raise RuntimeError("Empty vision response content.")
            return content
        except Exception as exc:
            last_exc = exc
            if idx == 0 and fallback_vision_model and _is_vision_unsupported_error(exc):
                logger.warning(
                    "Vision model '%s' may not support image input in current endpoint. "
                    "Falling back to '%s'.",
                    vision_model,
                    fallback_vision_model,
                )
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Vision description failed without explicit exception.")


def _extract_text_from_image_with_paddleocr(image_path: str) -> str:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is required for CPU-only OCR fallback. Install with: "
            ".\\.venv\\Scripts\\python.exe -m pip install paddleocr"
        ) from exc

    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(image_path, cls=True)
    lines: List[str] = []
    if isinstance(result, list):
        for block in result:
            if not isinstance(block, list):
                continue
            for item in block:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                    and len(item[1]) >= 1
                    and isinstance(item[1][0], str)
                ):
                    text = item[1][0].strip()
                    if text:
                        lines.append(text)
    return "\n".join(lines).strip()


def _doc_status_ok(working_dir: str, input_file_path: str) -> bool:
    """Check doc status is not failed for the current file."""
    status_path = Path(working_dir) / "kv_store_doc_status.json"
    if not status_path.exists():
        return False
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    target_name = Path(input_file_path).name.lower()
    for _, item in data.items():
        if not isinstance(item, dict):
            continue
        file_ref = str(item.get("file_path", "")).lower()
        if file_ref.endswith(target_name):
            return str(item.get("status", "")).lower() not in {"failed", "error"}
    return False


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
    parse_method: str = "auto",
    mineru_backend: str = "pipeline",
    mineru_device: str = "cpu",
    fallback_parser: Optional[str] = None,
    enable_parser_fallback: bool = True,
    max_chars: Optional[int] = None,
    max_paragraphs: Optional[int] = None,
    skip_query: bool = False,
    cli_queries: Optional[List[str]] = None,
    embedding_workers: int = 1,
    embedding_batch_num: int = 1,
    embedding_max_retries: int = 1,
    embedding_backoff_base_sec: float = 8.0,
    embedding_backoff_max_sec: float = 30.0,
    llm_max_retries: int = 0,
    llm_backoff_base_sec: float = 6.0,
    llm_backoff_max_sec: float = 20.0,
    pdf_mode: str = "auto",
    max_pages: Optional[int] = None,
    page_range: Optional[str] = None,
    vision_page_range: Optional[str] = None,
    max_vision_pages: Optional[int] = None,
    vision_target: Optional[str] = None,
    no_vision: bool = False,
    skip_kg_extraction: bool = False,
):
    try:
        parser_input = (parser or "mineru").strip().lower()
        parser_alias_pdf_hybrid = parser_input == "pdf_hybrid"
        if parser_alias_pdf_hybrid:
            parser = "pdf_fast"
            if (pdf_mode or "auto").lower() == "auto":
                pdf_mode = "hybrid"
            logger.info(
                "Parser alias pdf_hybrid resolved to pdf_fast with pdf_mode=hybrid"
            )

        abs_file_path = str(Path(file_path).resolve())
        abs_output_dir = str(Path(output_dir).resolve())
        abs_working_dir = str(Path(working_dir or "./rag_storage").resolve())
        abs_tmp_dir = str(Path("./.tmp").resolve())

        os.makedirs(abs_tmp_dir, exist_ok=True)
        os.makedirs(abs_output_dir, exist_ok=True)
        os.makedirs(abs_working_dir, exist_ok=True)

        llm_model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        vision_model = os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")
        fallback_vision_model = os.getenv(
            "FALLBACK_VISION_MODEL", "gemini-2.5-flash-lite"
        )
        llm_provider = _detect_provider_name(base_url)
        vision_provider = llm_provider

        async def llm_model_func(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ):
            attempts = max(0, int(llm_max_retries)) + 1
            delay = max(0.0, float(llm_backoff_base_sec))
            messages = kwargs.pop("messages", None)
            kwargs.pop("hashing_kv", None)
            kwargs.pop("keyword_extraction", None)
            kwargs.pop("enable_cot", None)
            kwargs.pop("response_format", None)
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(history_messages or [])
                messages.append({"role": "user", "content": prompt})

            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        max_retries=0,
                        timeout=180.0,
                    )
                    async with client:
                        resp = await client.chat.completions.create(
                            model=llm_model,
                            messages=messages,
                            **kwargs,
                        )
                    content = resp.choices[0].message.content if resp.choices else None
                    if content is None:
                        raise RuntimeError("Empty LLM response content.")
                    return content
                except RateLimitError as exc:
                    last_exc = exc
                    logger.error("Gemini quota exceeded (LLM).")
                    if attempt >= attempts:
                        raise
                    jitter = random.uniform(0, 0.5)
                    sleep_s = min(max(0.0, delay) * (1.0 + jitter), llm_backoff_max_sec)
                    logger.warning(
                        "LLM rate limited. Retrying in %.1fs (attempt %d/%d)...",
                        sleep_s,
                        attempt,
                        attempts,
                    )
                    await asyncio.sleep(sleep_s)
                    delay = min(max(1.0, delay) * 2.0, llm_backoff_max_sec)
                except (APITimeoutError, APIConnectionError) as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        raise
                    jitter = random.uniform(0, 0.5)
                    sleep_s = min(max(1.0, delay) * (1.0 + jitter), llm_backoff_max_sec)
                    logger.warning(
                        "LLM transient error. Retrying in %.1fs (attempt %d/%d): %s",
                        sleep_s,
                        attempt,
                        attempts,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_s)
                    delay = min(max(1.0, delay) * 2.0, llm_backoff_max_sec)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM call failed without explicit exception.")

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_provider = (
            os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_BINDING")
            or "openai"
        ).lower()
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        ollama_host = os.getenv("OLLAMA_HOST") or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

        async def gemini_embed_with_backoff(texts, model, api_key, base_url):
            attempts = max(0, int(embedding_max_retries)) + 1
            delay = max(0.0, float(embedding_backoff_base_sec))
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        max_retries=0,
                        timeout=60.0,
                    )
                    async with client:
                        resp = await client.embeddings.create(
                            model=model,
                            input=texts,
                            encoding_format="float",
                        )
                    return np.array([d.embedding for d in resp.data], dtype=np.float32)
                except RateLimitError as exc:
                    last_exc = exc
                    if _is_gemini_quota_exceeded(exc):
                        logger.error(
                            "Gemini quota exceeded during embeddings (429/RESOURCE_EXHAUSTED)."
                        )
                    if attempt >= attempts:
                        raise
                    jitter = random.uniform(0, 0.5)
                    sleep_s = min(
                        max(0.0, delay) * (1.0 + jitter), embedding_backoff_max_sec
                    )
                    logger.warning(
                        "Embedding rate limited. Retrying in %.1fs (attempt %d/%d)...",
                        sleep_s,
                        attempt,
                        attempts,
                    )
                    await asyncio.sleep(sleep_s)
                    delay = min(max(1.0, delay) * 2.0, embedding_backoff_max_sec)
                except (APITimeoutError, APIConnectionError) as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        raise
                    jitter = random.uniform(0, 0.5)
                    sleep_s = min(
                        max(1.0, delay) * (1.0 + jitter), embedding_backoff_max_sec
                    )
                    logger.warning(
                        "Embedding transient error. Retrying in %.1fs (attempt %d/%d): %s",
                        sleep_s,
                        attempt,
                        attempts,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_s)
                    delay = min(max(1.0, delay) * 2.0, embedding_backoff_max_sec)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Embedding failed without explicit exception.")

        async def ollama_embed_with_backoff(texts, model, host, target_dim):
            import aiohttp

            attempts = max(0, int(embedding_max_retries)) + 1
            delay = max(0.0, float(embedding_backoff_base_sec))
            last_exc = None
            url = f"{host.rstrip('/')}/api/embed"
            for attempt in range(1, attempts + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json={"model": model, "input": texts},
                            timeout=60,
                        ) as resp:
                            if resp.status != 200:
                                body = await resp.text()
                                raise RuntimeError(
                                    f"Ollama embed failed with status {resp.status}: {body[:300]}"
                                )
                            data = await resp.json()
                    emb = data.get("embeddings", [])
                    if not emb:
                        raise RuntimeError("Ollama returned empty embeddings.")
                    if len(emb[0]) != target_dim:
                        raise ValueError(
                            f"Embedding dim mismatch: expected {target_dim}, got {len(emb[0])}. "
                            "Check EMBEDDING_DIM/EMBEDDING_MODEL."
                        )
                    return np.array(emb, dtype=np.float32)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        raise
                    jitter = random.uniform(0, 0.5)
                    sleep_s = min(
                        max(1.0, delay) * (1.0 + jitter), embedding_backoff_max_sec
                    )
                    logger.warning(
                        "Ollama embedding transient error. Retrying in %.1fs (attempt %d/%d): %s",
                        sleep_s,
                        attempt,
                        attempts,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_s)
                    delay = min(max(1.0, delay) * 2.0, embedding_backoff_max_sec)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Ollama embedding failed without explicit exception.")

        if embedding_provider == "ollama":
            if not embedding_model:
                raise RuntimeError(
                    "Embedding provider/model mismatch: EMBEDDING_PROVIDER=ollama requires EMBEDDING_MODEL."
                )
            await _ensure_ollama_model_available(ollama_host, embedding_model)
            logger.info("Embedding provider: ollama")
            logger.info("Embedding model: %s", embedding_model)
            logger.info("Embedding dim: %s", embedding_dim)
            logger.info("Embedding host: %s", ollama_host)
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=partial(
                    ollama_embed_with_backoff,
                    model=embedding_model,
                    host=ollama_host,
                    target_dim=embedding_dim,
                ),
            )
        elif embedding_provider in {"gemini", "openai"}:
            if (
                embedding_provider == "gemini"
                and embedding_model == "text-embedding-3-large"
            ):
                raise RuntimeError(
                    "Embedding provider/model mismatch: Gemini provider cannot use text-embedding-3-large. "
                    "Use EMBEDDING_MODEL=gemini-embedding-001 or switch to EMBEDDING_PROVIDER=ollama."
                )
            logger.info("Embedding provider: %s", embedding_provider)
            logger.info("Embedding model: %s", embedding_model)
            logger.info("Embedding dim: %s", embedding_dim)
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=partial(
                    gemini_embed_with_backoff,
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
        else:
            raise RuntimeError(
                f"Unsupported embedding provider '{embedding_provider}'. "
                "Supported: ollama, gemini, openai."
            )

        logger.info("LLM provider: %s", llm_provider)
        logger.info("LLM model: %s", llm_model)
        logger.info("Vision provider: %s", vision_provider)
        logger.info("Vision model: %s", vision_model)

        env_for_parser = {"TEMP": abs_tmp_dir, "TMP": abs_tmp_dir}

        logger.info("Input file: %s", abs_file_path)
        logger.info("Output dir: %s", abs_output_dir)
        logger.info("Working dir: %s", abs_working_dir)
        logger.info("TEMP/TMP dir: %s", abs_tmp_dir)
        logger.info("Embedding workers: %s", embedding_workers)
        logger.info("Embedding batch size: %s", embedding_batch_num)
        logger.info("Embedding max retries: %s", embedding_max_retries)
        logger.info(
            "Embedding backoff base/max sec: %s/%s",
            embedding_backoff_base_sec,
            embedding_backoff_max_sec,
        )
        logger.info("LLM max retries: %s", llm_max_retries)
        logger.info(
            "LLM backoff base/max sec: %s/%s", llm_backoff_base_sec, llm_backoff_max_sec
        )
        if max_chars is not None:
            logger.info("simple_docx max_chars: %s", max_chars)
        if max_paragraphs is not None:
            logger.info("simple_docx max_paragraphs: %s", max_paragraphs)
        logger.info("skip_kg_extraction (best-effort): %s", skip_kg_extraction)
        logger.info("no_vision: %s", no_vision)
        if vision_page_range:
            logger.info("vision_page_range: %s", vision_page_range)
        if vision_target:
            logger.info("vision_target: %s", vision_target)
        if max_vision_pages:
            logger.info("max_vision_pages: %s", max_vision_pages)

        is_image_input = _is_image_file(abs_file_path)
        is_pdf_input = _is_pdf_file(abs_file_path)
        parser_order: List[str] = []
        primary_parser = (parser or "mineru").strip().lower()
        parser_order.append(primary_parser)

        if (
            enable_parser_fallback
            and (not parser_alias_pdf_hybrid)
            and primary_parser not in {"docling", "simple_docx"}
        ):
            fb = (fallback_parser or "").strip().lower()
            if fb and fb not in parser_order:
                parser_order.append(fb)
            if "docling" not in parser_order:
                parser_order.append("docling")

        # CPU-only policy: do not run MinerU for image flow
        parser_order = [p for p in parser_order if p != "mineru"]
        if not parser_order:
            parser_order = ["paddleocr", "docling"] if is_image_input else ["docling"]

        parse_success = False
        used_parser = None
        rag = None
        last_error = None

        pdf_mode_normalized = (pdf_mode or "auto").lower()
        use_pdf_fast = is_pdf_input and (
            primary_parser == "pdf_fast" or pdf_mode_normalized == "fast"
        )
        use_pdf_hybrid = is_pdf_input and (
            primary_parser == "pdf_fast" and pdf_mode_normalized == "hybrid"
        )

        if use_pdf_fast or use_pdf_hybrid:
            t_init = perf_counter()
            config = RAGAnythingConfig(
                working_dir=abs_working_dir,
                parser="docling",
                parse_method=parse_method,
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
                lightrag_kwargs={
                    "embedding_func_max_async": max(1, int(embedding_workers)),
                    "embedding_batch_num": max(1, int(embedding_batch_num)),
                    "llm_model_max_async": 1,
                    "max_parallel_insert": 1,
                    "entity_extract_max_gleaning": 0 if skip_kg_extraction else 1,
                },
            )
            logger.info(
                "PDF %s mode enabled (CPU text extraction)",
                "hybrid" if use_pdf_hybrid else "fast",
            )
            logger.info(
                "PDF fast options: max_pages=%s, page_range=%s", max_pages, page_range
            )
            logger.info("Parser init duration: %.2fs", perf_counter() - t_init)
            t_extract = perf_counter()
            fast_content = _extract_text_from_pdf_fast(
                abs_file_path,
                max_pages=max_pages,
                page_range=page_range,
            )
            logger.info(
                "PDF fast extraction duration: %.2fs", perf_counter() - t_extract
            )
            if not fast_content:
                logger.error("PDF fast extraction found no text content.")
                return
            combined_content = list(fast_content)
            if use_pdf_hybrid:
                t_table = perf_counter()
                table_blocks = _extract_tables_from_pdf_pdfplumber(
                    abs_file_path,
                    max_pages=max_pages,
                    page_range=page_range,
                )
                logger.info(
                    "pdfplumber table extraction duration: %.2fs",
                    perf_counter() - t_table,
                )
                combined_content.extend(table_blocks)

                if (not no_vision) and _has_vision_provider(
                    api_key=api_key, base_url=base_url
                ):
                    t_render = perf_counter()
                    render_dir = str(
                        Path(abs_tmp_dir) / "pdf_hybrid" / Path(abs_file_path).stem
                    )
                    effective_vision_page_range = vision_page_range
                    skip_render_due_to_target_not_found = False
                    if vision_page_range and vision_target:
                        logger.info(
                            "Both vision target and vision page range provided; using explicit --vision-page-range"
                        )
                    elif vision_target:
                        target_page_idx = _resolve_vision_target_page_from_text_blocks(
                            fast_content, vision_target
                        )
                        if target_page_idx is not None:
                            target_page_1based = target_page_idx + 1
                            effective_vision_page_range = str(target_page_1based)
                            logger.info(
                                "Vision target '%s' matched page %s",
                                vision_target,
                                target_page_1based,
                            )
                        else:
                            logger.warning(
                                "Vision target '%s' not found in text layer; use --vision-page-range to specify pages manually",
                                vision_target,
                            )
                            skip_render_due_to_target_not_found = True

                    if skip_render_due_to_target_not_found:
                        image_items = []
                    else:
                        image_items = _render_pdf_pages_for_vision(
                            abs_file_path,
                            tmp_dir=render_dir,
                            max_pages=max_vision_pages or max_pages,
                            page_range=effective_vision_page_range
                            or (None if vision_target else page_range),
                        )
                    logger.info(
                        "PDF page rendering duration: %.2fs",
                        perf_counter() - t_render,
                    )
                    if image_items:
                        t_vision = perf_counter()
                        quota_hit = False
                        described = 0
                        for item in image_items:
                            if quota_hit:
                                break
                            img_path = item.get("img_path")
                            page_idx = int(item.get("page_idx", 0))
                            try:
                                desc = await _describe_image_with_vision(
                                    image_path=img_path,
                                    prompt="Describe key visual elements, tables/charts if visible, and equations if any.",
                                    api_key=api_key,
                                    base_url=base_url,
                                    vision_model=vision_model,
                                    fallback_vision_model=fallback_vision_model,
                                )
                                combined_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[PDF Vision | page={page_idx + 1}] {desc}",
                                        "page_idx": page_idx,
                                        "source": "pdf_page_vision",
                                    }
                                )
                                described += 1
                            except Exception as vision_exc:
                                if _is_gemini_quota_exceeded(vision_exc):
                                    logger.error(
                                        "Gemini quota exceeded; skipped remaining visual descriptions"
                                    )
                                    quota_hit = True
                                    break
                                logger.warning(
                                    "Vision description failed for page %s: %s",
                                    page_idx + 1,
                                    str(vision_exc),
                                )
                        logger.info(
                            "PDF vision description duration: %.2fs (described %s/%s)",
                            perf_counter() - t_vision,
                            described,
                            len(image_items),
                        )
                else:
                    logger.info(
                        "Vision provider not available; skipped PDF page visual descriptions."
                    )

            t_insert = perf_counter()
            await rag.insert_content_list(combined_content, file_path=abs_file_path)
            logger.info("Content insertion duration: %.2fs", perf_counter() - t_insert)
            used_parser = "pdf_hybrid" if use_pdf_hybrid else "pdf_fast"
            parse_success = True

        if is_image_input:
            parser_name = (
                primary_parser
                if primary_parser in {"paddleocr", "docling"}
                else "paddleocr"
            )
            t_init = perf_counter()
            config = RAGAnythingConfig(
                working_dir=abs_working_dir,
                parser=parser_name,
                parse_method=parse_method,
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
                lightrag_kwargs={
                    "embedding_func_max_async": max(1, int(embedding_workers)),
                    "embedding_batch_num": max(1, int(embedding_batch_num)),
                    "llm_model_max_async": 1,
                    "max_parallel_insert": 1,
                    "entity_extract_max_gleaning": 0 if skip_kg_extraction else 1,
                },
            )
            logger.info("Parser init duration: %.2fs", perf_counter() - t_init)

            has_vision = _has_vision_provider(api_key=api_key, base_url=base_url)
            logger.info("Image input detected: %s", abs_file_path)
            logger.info("Vision provider available: %s", has_vision)
            try:
                if has_vision:
                    content_list = [
                        {
                            "type": "image",
                            "img_path": abs_file_path,
                            "image_caption": [],
                            "image_footnote": [],
                            "page_idx": 0,
                        }
                    ]
                    await rag.insert_content_list(content_list, file_path=abs_file_path)
                    used_parser = f"{parser_name}-vision"
                    parse_success = True
                else:
                    ocr_text = _extract_text_from_image_with_paddleocr(abs_file_path)
                    if not ocr_text:
                        logger.error(
                            "OCR found no text in image; visual description requires a vision model."
                        )
                        return
                    content_list = [{"type": "text", "text": ocr_text, "page_idx": 0}]
                    await rag.insert_content_list(content_list, file_path=abs_file_path)
                    used_parser = "paddleocr-ocr"
                    parse_success = True
            except Exception as image_exc:
                last_error = image_exc
                logger.warning(
                    "Image vision path failed: %s. Trying OCR fallback...",
                    str(image_exc),
                )
                try:
                    ocr_text = _extract_text_from_image_with_paddleocr(abs_file_path)
                    if not ocr_text:
                        logger.error(
                            "OCR found no text in image; visual description requires a vision model."
                        )
                        return
                    content_list = [{"type": "text", "text": ocr_text, "page_idx": 0}]
                    await rag.insert_content_list(content_list, file_path=abs_file_path)
                    used_parser = "paddleocr-ocr"
                    parse_success = True
                except Exception as ocr_exc:
                    last_error = ocr_exc

        for parser_name in parser_order:
            if parse_success:
                break
            try:
                t_init = perf_counter()
                config = RAGAnythingConfig(
                    working_dir=abs_working_dir,
                    parser=parser_name,
                    parse_method=parse_method,
                    enable_image_processing=True,
                    enable_table_processing=True,
                    enable_equation_processing=True,
                )
                rag = RAGAnything(
                    config=config,
                    llm_model_func=llm_model_func,
                    vision_model_func=vision_model_func,
                    embedding_func=embedding_func,
                    lightrag_kwargs={
                        "embedding_func_max_async": max(1, int(embedding_workers)),
                        "embedding_batch_num": max(1, int(embedding_batch_num)),
                        "llm_model_max_async": 1,
                        "max_parallel_insert": 1,
                        "entity_extract_max_gleaning": 0 if skip_kg_extraction else 1,
                    },
                )
                logger.info("Parser init duration: %.2fs", perf_counter() - t_init)

                logger.info("Parser trial: %s", parser_name)
                logger.info("Parse method: %s", parse_method)
                if is_pdf_input and parser_name == "docling":
                    logger.warning(
                        "Docling deep parsing on CPU can be slow for complex PDFs."
                    )
                parser_kwargs = {
                    "file_path": abs_file_path,
                    "output_dir": abs_output_dir,
                    "parse_method": parse_method,
                    "env": env_for_parser,
                    "allow_mineru_image_fallback": False,
                }
                if parser_name == "mineru":
                    logger.info("MinerU backend: %s", mineru_backend)
                    logger.info("MinerU device: %s", mineru_device)
                    parser_kwargs["backend"] = mineru_backend
                    parser_kwargs["device"] = mineru_device
                # simple_docx-only trimming options; do not pass to image/parser paths
                if (
                    parser_name == "simple_docx"
                    and Path(abs_file_path).suffix.lower() == ".docx"
                ):
                    parser_kwargs["max_chars"] = max_chars
                    parser_kwargs["max_paragraphs"] = max_paragraphs
                t_parse = perf_counter()
                await rag.process_document_complete(**parser_kwargs)
                logger.info(
                    "Document parse/index duration: %.2fs", perf_counter() - t_parse
                )

                if not _is_index_ready(abs_working_dir):
                    raise RuntimeError(
                        "Index artifacts were not created in working_dir. "
                        "Index step likely failed."
                    )
                if not _doc_status_ok(abs_working_dir, abs_file_path):
                    raise RuntimeError(
                        "Document index status is failed in kv_store_doc_status.json."
                    )
                parse_success = True
                used_parser = parser_name
                try:
                    status_path = Path(abs_working_dir) / "kv_store_parse_cache.json"
                    if status_path.exists():
                        logger.debug("Parse cache file exists: %s", status_path)
                except Exception:
                    pass
                logger.info(
                    "Document parsed/indexed successfully with parser: %s", used_parser
                )
                if is_pdf_input and parser_name == "docling":
                    logger.warning(
                        "Docling extracted text-only content; multimodal elements were not separated."
                    )
                break
            except Exception as e:
                last_error = e
                if _is_gemini_quota_exceeded(e):
                    logger.error(
                        "Gemini quota exceeded. Stop early to avoid long retries."
                    )
                    break
                if (
                    parser_name == "mineru"
                    and _is_windows_winerror5(e)
                    and enable_parser_fallback
                ):
                    logger.warning(
                        "MinerU failed on Windows CPU due to WinError 5. Trying fallback parser..."
                    )
                    continue

                if enable_parser_fallback and parser_name != parser_order[-1]:
                    logger.warning(
                        "Parser '%s' failed: %s. Trying next fallback...",
                        parser_name,
                        str(e),
                    )
                    continue

                break

        if not parse_success or rag is None:
            if cli_queries:
                logger.error("Document processing failed, cannot answer user query")
            else:
                logger.error("Document processing failed. Query step skipped.")
            if last_error is not None:
                logger.error("Last parser error: %s", str(last_error))
            logger.error(
                "Check dependencies/API config. If parser missing, install one of these:\n"
                "1) .\\.venv\\Scripts\\python.exe -m pip install paddleocr pypdfium2\n"
                "2) .\\.venv\\Scripts\\python.exe -m pip install docling\n"
                "If embedding fails on Gemini, set EMBEDDING_MODEL=gemini-embedding-001 and EMBEDDING_DIM=3072."
            )
            return

        if skip_query:
            logger.info(
                "Skip query enabled. Ingest/index completed, query step skipped."
            )
            return

        logger.info("\nQuerying processed document (parser used: %s):", used_parser)

        # If CLI provided queries (via --query / -q), use them; otherwise use default sample queries
        if cli_queries and len(cli_queries) > 0:
            text_queries = cli_queries
        else:
            text_queries = [
                "Trong Bộ luật Lao động 2012, Điều 1 quy định về nội dung gì?",
                "Trong Bộ luật Lao động 2019, độ tuổi lao động tối thiểu là bao nhiêu?",
                "Khái niệm cưỡng bức lao động được định nghĩa ở đâu trong Bộ luật Lao động 2019?",
                "Bộ luật Lao động 2012 có áp dụng cho người lao động nước ngoài làm việc tại Việt Nam không?",
                "So sánh phạm vi điều chỉnh của Bộ luật Lao động 2012 và Bộ luật Lao động 2019.",
            ]

        for query in text_queries:
            t_query = perf_counter()
            logger.info("\n[Text Query]: %s", query)
            try:
                result = await rag.aquery(query, mode="hybrid")
            except Exception as query_exc:
                if _is_gemini_quota_exceeded(query_exc):
                    logger.error(
                        "Gemini quota exceeded during query. Stopping query loop."
                    )
                    return
                if "expected string or bytes-like object" in str(query_exc):
                    logger.error(
                        "Query failed because LLM returned empty content after upstream failure."
                    )
                    return
                raise
            if result is None:
                logger.error("Query returned no answer. Stopping remaining queries.")
                return
            logger.info("Answer: %s", result)
            logger.info("Query duration: %.2fs", perf_counter() - t_query)

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI-compatible API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Parser selection. Built-ins: mineru, docling, paddleocr, simple_docx, pdf_fast. Alias: pdf_hybrid -> pdf_fast + pdf_mode=hybrid.",
    )
    parser.add_argument(
        "--parse-method",
        default=os.getenv("PARSE_METHOD", "auto"),
        help="Parse method for parser (auto, ocr, txt).",
    )
    parser.add_argument(
        "--mineru-backend",
        default=os.getenv("MINERU_BACKEND", "pipeline"),
        help="MinerU backend (default: pipeline for CPU-safe execution).",
    )
    parser.add_argument(
        "--mineru-device",
        default=os.getenv("MINERU_DEVICE", "cpu"),
        help="MinerU device (default: cpu).",
    )
    parser.add_argument(
        "--fallback-parser",
        default=os.getenv("FALLBACK_PARSER", "paddleocr"),
        help="Fallback parser when primary parser fails (default: paddleocr).",
    )
    parser.add_argument(
        "--disable-parser-fallback",
        action="store_true",
        help="Disable parser fallback behavior.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("SIMPLE_DOCX_MAX_CHARS", "0")) or None,
        help="For parser simple_docx: keep only first N characters after conversion (0 = no limit).",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=int(os.getenv("SIMPLE_DOCX_MAX_PARAGRAPHS", "0")) or None,
        help="For parser simple_docx: keep only first N non-empty paragraphs (0 = no limit).",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Ingest/index only. Skip query step.",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=int(os.getenv("EMBEDDING_WORKERS", "1")),
        help="Max async embedding workers (default: 1).",
    )
    parser.add_argument(
        "--embedding-batch-num",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_NUM", "1")),
        help="Embedding batch size for LightRAG insert (default: 1).",
    )
    parser.add_argument(
        "--embedding-max-retries",
        type=int,
        default=int(os.getenv("EMBEDDING_MAX_RETRIES", "1")),
        help="Embedding retries on transient errors (default: 1).",
    )
    parser.add_argument(
        "--embedding-backoff-base-sec",
        type=float,
        default=float(os.getenv("EMBEDDING_BACKOFF_BASE_SEC", "8")),
        help="Base backoff seconds for embedding retries (default: 8).",
    )
    parser.add_argument(
        "--embedding-backoff-max-sec",
        type=float,
        default=float(os.getenv("EMBEDDING_BACKOFF_MAX_SEC", "30")),
        help="Max backoff seconds for embedding retries (default: 30).",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=int(os.getenv("LLM_MAX_RETRIES", "0")),
        help="LLM retries on transient errors (default: 0 for fail-fast quota handling).",
    )
    parser.add_argument(
        "--llm-backoff-base-sec",
        type=float,
        default=float(os.getenv("LLM_BACKOFF_BASE_SEC", "6")),
        help="Base backoff seconds for LLM retries (default: 6).",
    )
    parser.add_argument(
        "--llm-backoff-max-sec",
        type=float,
        default=float(os.getenv("LLM_BACKOFF_MAX_SEC", "20")),
        help="Max backoff seconds for LLM retries (default: 20).",
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Text query to run after ingestion. Can be specified multiple times.",
    )
    parser.add_argument(
        "--pdf-mode",
        default=os.getenv("PDF_MODE", "auto"),
        choices=["auto", "fast", "deep", "hybrid"],
        help="PDF processing mode: fast (text extraction), hybrid (text+tables+optional rendered-page vision), deep (docling), auto (default parser behavior).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=int(os.getenv("PDF_MAX_PAGES", "0")) or None,
        help="For PDF fast mode: limit number of pages to extract.",
    )
    parser.add_argument(
        "--page-range",
        default=os.getenv("PDF_PAGE_RANGE"),
        help="For PDF fast/hybrid mode: page selection like 3,5,7-13.",
    )
    parser.add_argument(
        "--vision-page-range",
        default=os.getenv("VISION_PAGE_RANGE"),
        help="For PDF hybrid mode: page selection used only for vision rendering, e.g. 3,5,7-9.",
    )
    parser.add_argument(
        "--max-vision-pages",
        type=int,
        default=int(os.getenv("MAX_VISION_PAGES", "0")) or None,
        help="For PDF hybrid mode: cap rendered pages sent to vision model.",
    )
    parser.add_argument(
        "--vision-target",
        default=os.getenv("VISION_TARGET"),
        help='For PDF hybrid mode: auto-resolve target mention to vision page, e.g. --vision-target "Fig. 2".',
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision descriptions (text/table local path only).",
    )
    parser.add_argument(
        "--skip-kg-extraction",
        "--vector-only",
        action="store_true",
        help="Best-effort quota-saving: reduce KG extraction workload.",
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("Error: API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
            args.parse_method,
            args.mineru_backend,
            args.mineru_device,
            args.fallback_parser,
            (not args.disable_parser_fallback)
            and os.getenv("ENABLE_PARSER_FALLBACK", "true").lower() == "true",
            args.max_chars,
            args.max_paragraphs,
            args.skip_query,
            args.query,
            args.embedding_workers,
            args.embedding_batch_num,
            args.embedding_max_retries,
            args.embedding_backoff_base_sec,
            args.embedding_backoff_max_sec,
            args.llm_max_retries,
            args.llm_backoff_base_sec,
            args.llm_backoff_max_sec,
            args.pdf_mode,
            args.max_pages,
            args.page_range,
            args.vision_page_range,
            args.max_vision_pages,
            args.vision_target,
            args.no_vision,
            args.skip_kg_extraction,
        )
    )


if __name__ == "__main__":
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
