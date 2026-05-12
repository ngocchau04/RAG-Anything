#!/usr/bin/env python
"""
Example script demonstrating parser integration with RAGAnything
"""

import os
import json
import argparse
import asyncio
import logging
import logging.config
from functools import partial
from pathlib import Path
from typing import Optional, List
import random
import numpy as np

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
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
    embedding_workers: int = 1,
    embedding_batch_num: int = 1,
    embedding_max_retries: int = 1,
    embedding_backoff_base_sec: float = 8.0,
    embedding_backoff_max_sec: float = 30.0,
    llm_max_retries: int = 0,
    llm_backoff_base_sec: float = 6.0,
    llm_backoff_max_sec: float = 20.0,
):
    try:
        abs_file_path = str(Path(file_path).resolve())
        abs_output_dir = str(Path(output_dir).resolve())
        abs_working_dir = str(Path(working_dir or "./rag_storage").resolve())
        abs_tmp_dir = str(Path("./.tmp").resolve())

        os.makedirs(abs_tmp_dir, exist_ok=True)
        os.makedirs(abs_output_dir, exist_ok=True)
        os.makedirs(abs_working_dir, exist_ok=True)

        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        vision_model = os.getenv("VISION_MODEL", "gpt-4o")

        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
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

        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-local")

        async def ollama_embed_local(texts, model, base_url, target_dim):
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{base_url.rstrip('/')}/api/tags") as tags_resp:
                            if tags_resp.status != 200:
                                raise Exception("Server error on /api/tags")
                            tags_data = await tags_resp.json()
                            models = [m["name"] for m in tags_data.get("models", [])]
                            model_names = [m.split(":")[0] for m in models] + models
                            if model not in model_names:
                                raise ValueError(f"Ollama model not found. Current model should be {model}.")
                    except ValueError:
                        raise
                    except Exception:
                        raise ConnectionError("Ollama server is not running. Please start Ollama.")

                    url = f"{base_url.rstrip('/')}/api/embed"
                    payload = {"model": model, "input": texts}
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if "embeddings" in data:
                                emb = data["embeddings"]
                                if len(emb) > 0 and len(emb[0]) != target_dim:
                                    raise ValueError(f"Ollama embedding dimension mismatch. Expected {target_dim}, got {len(emb[0])}")
                                return np.array(emb, dtype=np.float32)

                    # Fallback to /api/embeddings
                    url = f"{base_url.rstrip('/')}/api/embeddings"
                    results = []
                    for text in texts:
                        payload = {"model": model, "prompt": text}
                        async with session.post(url, json=payload) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                emb = data["embedding"]
                                if len(emb) != target_dim:
                                    raise ValueError(f"Ollama embedding dimension mismatch. Expected {target_dim}, got {len(emb)}")
                                results.append(emb)
                            else:
                                raise RuntimeError(f"Ollama embedding failed with status {resp.status}")
                    return np.array(results, dtype=np.float32)
            except aiohttp.ClientError:
                raise ConnectionError("Ollama server is not running. Please start Ollama.")

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
                    sleep_s = min(max(0.0, delay) * (1.0 + jitter), embedding_backoff_max_sec)
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
                    sleep_s = min(max(1.0, delay) * (1.0 + jitter), embedding_backoff_max_sec)
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

        if embedding_provider == "ollama":
            logger.info("EMBEDDING_PROVIDER=%s", embedding_provider)
            logger.info("OLLAMA_BASE_URL=%s", ollama_base_url)
            logger.info("OLLAMA_EMBEDDING_MODEL=%s", ollama_embedding_model)
            logger.info("EMBEDDING_DIM=%s", embedding_dim)
            logger.info("LLM_MODEL=%s", llm_model)

            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=partial(
                    ollama_embed_local,
                    model=ollama_embedding_model,
                    base_url=ollama_base_url,
                    target_dim=embedding_dim,
                ),
            )
        else:
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

        env_for_parser = {"TEMP": abs_tmp_dir, "TMP": abs_tmp_dir}

        logger.info("Input file: %s", abs_file_path)
        logger.info("Output dir: %s", abs_output_dir)
        logger.info("Working dir: %s", abs_working_dir)
        logger.info("TEMP/TMP dir: %s", abs_tmp_dir)
        logger.info("Embedding workers: %s", embedding_workers)
        logger.info("Embedding batch size: %s", embedding_batch_num)
        logger.info("Embedding max retries: %s", embedding_max_retries)
        logger.info("Embedding backoff base/max sec: %s/%s", embedding_backoff_base_sec, embedding_backoff_max_sec)
        logger.info("LLM max retries: %s", llm_max_retries)
        logger.info("LLM backoff base/max sec: %s/%s", llm_backoff_base_sec, llm_backoff_max_sec)
        if max_chars is not None:
            logger.info("simple_docx max_chars: %s", max_chars)
        if max_paragraphs is not None:
            logger.info("simple_docx max_paragraphs: %s", max_paragraphs)

        parser_order: List[str] = []
        primary_parser = (parser or "mineru").strip().lower()
        parser_order.append(primary_parser)

        if enable_parser_fallback and primary_parser not in {"docling", "simple_docx"}:
            fb = (fallback_parser or "").strip().lower()
            if fb and fb not in parser_order:
                parser_order.append(fb)
            if "docling" not in parser_order:
                parser_order.append("docling")

        parse_success = False
        used_parser = None
        rag = None
        last_error = None

        for parser_name in parser_order:
            try:
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
                    },
                )

                logger.info("Parser trial: %s", parser_name)
                logger.info("Parse method: %s", parse_method)
                if parser_name == "mineru":
                    logger.info("MinerU backend: %s", mineru_backend)
                    logger.info("MinerU device: %s", mineru_device)
                    await rag.process_document_complete(
                        file_path=abs_file_path,
                        output_dir=abs_output_dir,
                        parse_method=parse_method,
                        backend=mineru_backend,
                        device=mineru_device,
                        env=env_for_parser,
                        max_chars=max_chars,
                        max_paragraphs=max_paragraphs,
                    )
                else:
                    await rag.process_document_complete(
                        file_path=abs_file_path,
                        output_dir=abs_output_dir,
                        parse_method=parse_method,
                        env=env_for_parser,
                        max_chars=max_chars,
                        max_paragraphs=max_paragraphs,
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
                logger.info(
                    "Document parsed/indexed successfully with parser: %s", used_parser
                )
                break
            except Exception as e:
                last_error = e
                if _is_gemini_quota_exceeded(e):
                    logger.error("Gemini quota exceeded. Stop early to avoid long retries.")
                    break
                if parser_name == "mineru" and _is_windows_winerror5(e) and enable_parser_fallback:
                    logger.warning(
                        "MinerU failed on Windows CPU due to WinError 5. Trying fallback parser..."
                    )
                    continue

                if enable_parser_fallback and parser_name != parser_order[-1]:
                    logger.warning("Parser '%s' failed: %s. Trying next fallback...", parser_name, str(e))
                    continue

                break

        if not parse_success or rag is None:
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
            logger.info("Skip query enabled. Ingest/index completed, query step skipped.")
            return

        logger.info("\nQuerying processed document (parser used: %s):", used_parser)

        text_queries = [
            "Trong Bộ luật Lao động 2012, Điều 1 quy định về nội dung gì?",
            "Trong Bộ luật Lao động 2019, độ tuổi lao động tối thiểu là bao nhiêu?",
            "Khái niệm cưỡng bức lao động được định nghĩa ở đâu trong Bộ luật Lao động 2019?",
            "Bộ luật Lao động 2012 có áp dụng cho người lao động nước ngoài làm việc tại Việt Nam không?",
            "So sánh phạm vi điều chỉnh của Bộ luật Lao động 2012 và Bộ luật Lao động 2019.",
        ]

        for query in text_queries:
            logger.info("\n[Text Query]: %s", query)
            try:
                result = await rag.aquery(query, mode="hybrid")
            except Exception as query_exc:
                if _is_gemini_quota_exceeded(query_exc):
                    logger.error("Gemini quota exceeded during query. Stopping query loop.")
                    return
                raise
            if result is None:
                logger.error("Query returned no answer. Stopping remaining queries.")
                return
            logger.info("Answer: %s", result)

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--working_dir", "-w", default="./rag_storage", help="Working directory path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory path")
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
        help="Parser selection. Built-ins: mineru, docling, paddleocr, simple_docx.",
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
            args.embedding_workers,
            args.embedding_batch_num,
            args.embedding_max_retries,
            args.embedding_backoff_base_sec,
            args.embedding_backoff_max_sec,
            args.llm_max_retries,
            args.llm_backoff_base_sec,
            args.llm_backoff_max_sec,
        )
    )


if __name__ == "__main__":
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
