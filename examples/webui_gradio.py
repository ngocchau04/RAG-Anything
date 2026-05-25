#!/usr/bin/env python
"""
Local Gradio WebUI for RAG-Anything (CPU-only friendly).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from threading import Thread
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc, logger
from openai import AsyncOpenAI

from raganything import RAGAnything, RAGAnythingConfig
from raganything.config import resolve_embedding_runtime_config

load_dotenv(dotenv_path=".env", override=False)

REGISTRY_PATH = Path("./rag_storage/webui_registry.json").resolve()
UPLOADS_DIR = Path("./rag_storage/webui_uploads").resolve()
DOCS_ROOT = Path("./rag_storage/webui_docs").resolve()
SUPPORTED_QUERY_LOAD_PARSERS = {"mineru", "docling", "paddleocr", "simple_docx"}


def _is_gemini_quota_exceeded(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
        or "rate limit" in msg.lower()
    )


def _is_temporary_vlm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in [
            "error code: 503",
            "unavailable",
            "high demand",
            "resource_exhausted",
            "error code: 429",
            "rate limit",
            "quota",
            "timed out",
            "timeout",
        ]
    )


def detect_parser_for_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf_hybrid"
    if suffix == ".docx":
        return "simple_docx"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "paddleocr"
    if suffix == ".pptx":
        return "docling"
    if suffix in {".md", ".html", ".htm"}:
        return "docling"
    return "docling"


def _is_index_ready(working_dir: str) -> bool:
    wd = Path(working_dir)
    expected = [
        wd / "graph_chunk_entity_relation.graphml",
        wd / "vdb_chunks.json",
        wd / "kv_store_text_chunks.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in expected)


def _normalize_to_messages(history):
    normalized = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content is not None:
                normalized.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            normalized.append({"role": "user", "content": str(user_msg)})
            normalized.append({"role": "assistant", "content": str(assistant_msg)})
    return normalized


def _append_messages(history, user_q: str, answer: str):
    messages = _normalize_to_messages(history)
    messages.append({"role": "user", "content": str(user_q)})
    messages.append({"role": "assistant", "content": str(answer)})
    return messages


def _safe_filename(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return out or "uploaded_file"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_candidate_filenames(question: str) -> set[str]:
    q = (question or "").lower()
    return set(
        re.findall(
            r"[a-zA-Z0-9_.-]+\.(?:pdf|docx|png|jpg|jpeg|pptx|md|html|htm)",
            q,
        )
    )


async def _ollama_embed_runtime(
    texts,
    *,
    model: str,
    host: str,
    target_dim: int,
):
    import aiohttp

    url = f"{host.rstrip('/')}/api/embed"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"model": model, "input": texts},
            timeout=60,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Ollama embed failed ({resp.status}): {body[:240]}")
            data = await resp.json()
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama returned empty embeddings.")
    if len(embeddings[0]) != target_dim:
        raise RuntimeError(
            f"Embedding dim mismatch. Expected {target_dim}, got {len(embeddings[0])}."
        )
    return np.array(embeddings, dtype=np.float32)


async def _llm_call_runtime(
    prompt,
    *,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    system_prompt=None,
    history_messages=None,
    **kwargs,
):
    history_messages = history_messages or []
    messages = kwargs.pop("messages", None)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    kwargs.pop("response_format", None)

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=180.0,
    )
    async with client:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
    content = resp.choices[0].message.content if resp.choices else None
    if content is None:
        raise RuntimeError("Empty LLM response content.")
    return content


async def _vision_call_runtime(
    prompt,
    *,
    vision_model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    llm_model: str,
    system_prompt=None,
    history_messages=None,
    image_data=None,
    messages=None,
    **kwargs,
):
    history_messages = history_messages or []
    if messages:
        return await _llm_call_runtime(
            "",
            model=vision_model,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            **kwargs,
        )

    if image_data:
        payload = [
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            },
        ]
        payload = [m for m in payload if m]
        return await _llm_call_runtime(
            "",
            model=vision_model,
            api_key=api_key,
            base_url=base_url,
            messages=payload,
            **kwargs,
        )

    return await _llm_call_runtime(
        prompt,
        model=llm_model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@dataclass
class DocumentRecord:
    doc_id: str
    original_filename: str
    stored_file_path: str
    working_dir: str
    file_type: str
    parser: str
    pdf_mode: str
    indexed_at: str
    status: str
    summary: str
    sha256: str
    source_metadata: dict[str, Any]
    visual_targets: list[dict[str, Any]]


class RegistryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[DocumentRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [DocumentRecord(**item) for item in raw]
        except Exception:
            logger.warning("Registry file is invalid. Starting with empty registry.")
            return []

    def save(self, records: list[DocumentRecord]) -> None:
        payload = [asdict(r) for r in records]
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


@dataclass
class UIState:
    processed: bool = False
    processing_summary: str = ""
    last_process_ts: float = 0.0
    ingest_count: int = 0
    query_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class WebUIRAGService:
    def __init__(self):
        self.state = UIState()
        self.registry_store = RegistryStore(REGISTRY_PATH)
        self.registry: list[DocumentRecord] = self.registry_store.load()
        self.rag_cache: dict[str, RAGAnything] = {}

        embedding_cfg = resolve_embedding_runtime_config(default_provider="ollama")
        self.embedding_provider = embedding_cfg.provider
        self.embedding_model = embedding_cfg.model
        self.embedding_dim = embedding_cfg.dim
        self.ollama_host = embedding_cfg.ollama_host
        self.embedding_cfg = embedding_cfg

        self.llm_model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        self.llm_model_source = (
            "env:LLM_MODEL" if os.getenv("LLM_MODEL") else "default:gemini-3.1-flash-lite"
        )
        self.vision_model = os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")
        self.vision_model_source = (
            "env:VISION_MODEL"
            if os.getenv("VISION_MODEL")
            else "default:gemini-3.1-flash-lite"
        )
        self.fallback_vision_model = os.getenv(
            "FALLBACK_VISION_MODEL", "gemini-2.5-flash-lite"
        )
        self.fallback_vision_model_source = (
            "env:FALLBACK_VISION_MODEL"
            if os.getenv("FALLBACK_VISION_MODEL")
            else "default:gemini-2.5-flash-lite"
        )
        self.api_key = (
            os.getenv("LLM_BINDING_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.base_url = os.getenv("LLM_BINDING_HOST")

        self._loop = asyncio.new_event_loop()
        self._loop_thread = Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        logger.info("Embedding provider: %s", self.embedding_provider)
        logger.info("Embedding model: %s", self.embedding_model)
        logger.info("Embedding dim: %s", self.embedding_dim)
        logger.info("Embedding host: %s", self.ollama_host)
        logger.info(
            "Embedding config source: provider=%s, model=%s, dim=%s, host=%s",
            self.embedding_cfg.provider_source,
            self.embedding_cfg.model_source,
            self.embedding_cfg.dim_source,
            self.embedding_cfg.host_source,
        )
        logger.info("LLM model: %s", self.llm_model)
        logger.info("LLM model source: %s", self.llm_model_source)
        logger.info("Vision model: %s", self.vision_model)
        logger.info("Vision model source: %s", self.vision_model_source)
        logger.info("Fallback vision model: %s", self.fallback_vision_model)
        logger.info(
            "Fallback vision model source: %s", self.fallback_vision_model_source
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.registry)

    def _reload_registry(self) -> None:
        # Keep in-memory registry consistent with persisted state across callbacks.
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

    def _persist_upload_file(self, file_path: str, doc_id: str) -> str:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(file_path)
        dst = UPLOADS_DIR / f"{doc_id}_{_safe_filename(src.name)}"
        shutil.copy2(src, dst)
        return str(dst.resolve())

    @contextmanager
    def _patched_env_for_ingest(self):
        keys = {
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_BINDING": "ollama",
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_DIM": str(self.embedding_dim),
            "OLLAMA_HOST": self.ollama_host,
        }
        old = {k: os.environ.get(k) for k in keys}
        try:
            for k, v in keys.items():
                os.environ[k] = v
            yield
        finally:
            for k, old_v in old.items():
                if old_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_v

    async def _ensure_ollama_model_available(self) -> None:
        import aiohttp

        if self.embedding_provider != "ollama":
            return
        url = f"{self.ollama_host.rstrip('/')}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Ollama host check failed ({resp.status}) at {url}"
                    )
                data = await resp.json()
        names = [
            m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)
        ]
        aliases = {n for n in names}
        aliases.update({n.split(":")[0] for n in names})
        if self.embedding_model not in aliases:
            raise RuntimeError(
                "Ollama model not found. Run ollama list or set EMBEDDING_MODEL to an existing model."
            )

    async def _create_rag(self, working_dir: str, parser: str) -> RAGAnything:
        if self.embedding_provider != "ollama":
            raise RuntimeError(
                "WebUI defaults to local embeddings only. Set EMBEDDING_PROVIDER=ollama."
            )
        await self._ensure_ollama_model_available()
        llm_model = self.llm_model
        vision_model = self.vision_model
        api_key = self.api_key
        base_url = self.base_url
        embedding_model = self.embedding_model
        ollama_host = self.ollama_host
        embedding_dim = self.embedding_dim

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=partial(
                _ollama_embed_runtime,
                model=embedding_model,
                host=ollama_host,
                target_dim=embedding_dim,
            ),
        )
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        return RAGAnything(
            config=config,
            llm_model_func=partial(
                _llm_call_runtime,
                model=llm_model,
                api_key=api_key,
                base_url=base_url,
            ),
            vision_model_func=partial(
                _vision_call_runtime,
                vision_model=vision_model,
                llm_model=llm_model,
                api_key=api_key,
                base_url=base_url,
            ),
            embedding_func=embedding_func,
            lightrag_kwargs={
                "embedding_func_max_async": 1,
                "embedding_batch_num": 1,
                "llm_model_max_async": 1,
                "max_parallel_insert": 1,
            },
        )

    async def _process_single_document(
        self,
        file_path: str,
        force_parser: Optional[str] = None,
        pdf_page_range: Optional[str] = None,
        vision_target: Optional[str] = None,
        vision_page_range: Optional[str] = None,
        max_vision_pages: int = 1,
        no_vision: bool = True,
        skip_kg_extraction: bool = True,
    ) -> str:
        from examples import raganything_example as rx

        src = Path(file_path)
        if not src.exists():
            return f"Missing file: {file_path}"

        parser = (force_parser or detect_parser_for_file(str(src))).lower()
        sha256 = _sha256_file(str(src))
        existing = self._find_by_hash(sha256)
        if existing and _is_index_ready(existing.working_dir):
            return (
                f"Skipped duplicate: {existing.original_filename} "
                f"(already indexed as {existing.doc_id})"
            )

        doc_id = existing.doc_id if existing else uuid.uuid4().hex[:12]
        stored_path = self._persist_upload_file(str(src), doc_id)
        workdir = DOCS_ROOT / doc_id
        workdir.mkdir(parents=True, exist_ok=True)

        pdf_mode = "auto"
        if parser == "pdf_hybrid":
            parser = "pdf_fast"
            pdf_mode = "hybrid"
            logger.info(
                "Parser alias pdf_hybrid resolved to pdf_fast with pdf_mode=hybrid"
            )
        if parser == "pdf_fast":
            pdf_mode = "hybrid"

        with self._patched_env_for_ingest():
            await rx.process_with_rag(
                file_path=stored_path,
                output_dir="./output",
                api_key=self.api_key,
                base_url=self.base_url,
                working_dir=str(workdir.resolve()),
                parser=(
                    "pdf_hybrid"
                    if parser == "pdf_fast" and pdf_mode == "hybrid"
                    else parser
                ),
                parse_method="auto",
                mineru_backend="pipeline",
                mineru_device="cpu",
                fallback_parser="docling",
                enable_parser_fallback=False,
                max_chars=None,
                max_paragraphs=None,
                skip_query=True,
                cli_queries=None,
                embedding_workers=1,
                embedding_batch_num=1,
                embedding_max_retries=1,
                embedding_backoff_base_sec=8.0,
                embedding_backoff_max_sec=30.0,
                # WebUI indexing should tolerate temporary Gemini 503 spikes.
                llm_max_retries=2,
                llm_backoff_base_sec=6.0,
                llm_backoff_max_sec=30.0,
                pdf_mode=pdf_mode,
                max_pages=None,
                page_range=pdf_page_range,
                vision_page_range=vision_page_range,
                max_vision_pages=max_vision_pages,
                vision_target=vision_target,
                no_vision=no_vision,
                skip_kg_extraction=skip_kg_extraction,
            )

        if not _is_index_ready(str(workdir.resolve())):
            return (
                f"Index for {src.name} is missing or invalid. "
                "Please reprocess this file."
            )

        record = DocumentRecord(
            doc_id=doc_id,
            original_filename=src.name,
            stored_file_path=stored_path,
            working_dir=str(workdir.resolve()),
            file_type=src.suffix.lower(),
            parser="pdf_hybrid" if parser == "pdf_fast" else parser,
            pdf_mode=pdf_mode,
            indexed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            status="indexed",
            summary="",
            sha256=sha256,
            source_metadata={
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
            },
            visual_targets=(existing.visual_targets if existing else []),
        )

        if existing:
            self.registry = [r for r in self.registry if r.doc_id != existing.doc_id]
        self.registry.append(record)
        self.registry_store.save(self.registry)
        self.rag_cache.pop(doc_id, None)

        note = ""
        if record.file_type in {".png", ".jpg", ".jpeg"} and no_vision:
            note = (
                "\nno_vision applies to PDF visual page rendering only; "
                "direct image input still uses vision."
            )
        return f"Indexed: {src.name} [{doc_id}]\nWorking dir: {workdir}{note}"

    async def process_index(
        self,
        file_paths: list[str] | str,
        force_parser: Optional[str] = None,
        pdf_page_range: Optional[str] = None,
        vision_target: Optional[str] = None,
        vision_page_range: Optional[str] = None,
        max_vision_pages: int = 1,
        no_vision: bool = True,
        skip_kg_extraction: bool = True,
    ) -> str:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if not file_paths:
            return "Please upload/select files first."
        if not self.api_key:
            return (
                "Missing API key. Set LLM_BINDING_API_KEY "
                "or GEMINI_API_KEY/GOOGLE_API_KEY."
            )

        t0 = time.perf_counter()
        logs = []
        for fp in file_paths:
            try:
                logs.append(
                    await self._process_single_document(
                        file_path=fp,
                        force_parser=force_parser,
                        pdf_page_range=pdf_page_range,
                        vision_target=vision_target,
                        vision_page_range=vision_page_range,
                        max_vision_pages=max_vision_pages,
                        no_vision=no_vision,
                        skip_kg_extraction=skip_kg_extraction,
                    )
                )
            except Exception as exc:
                if _is_gemini_quota_exceeded(exc):
                    logs.append(
                        "Gemini quota exceeded. Continue with text/table-only mode "
                        "or wait for quota reset."
                    )
                else:
                    logs.append(f"Failed indexing {Path(fp).name}: {exc}")

        self.state.processed = any(
            ("Indexed:" in x) or ("Skipped duplicate:" in x) for x in logs
        )
        self.state.last_process_ts = time.time()
        self.state.ingest_count += len(file_paths)
        elapsed = time.perf_counter() - t0
        self.state.processing_summary = (
            "\n\n".join(logs) + f"\n\nTotal processing time: {elapsed:.2f}s"
        )
        return self.state.processing_summary

    def route_query_to_documents(
        self, question: str, selected_doc_id: Optional[str] = None
    ) -> tuple[list[DocumentRecord], Optional[str]]:
        self._reload_registry()
        docs = [r for r in self.registry if r.status == "indexed"]
        if not docs:
            return [], "Please upload and process files first."

        q = (question or "").strip().lower()
        if not q:
            return [], "Please enter a question."

        # Highest priority: explicit filename mention in question.
        mentions = _extract_candidate_filenames(q)
        if mentions:
            matched = [d for d in docs if d.original_filename.lower() in mentions]
            if matched:
                return matched, None

        # Next priority: stem mention in question.
        stem_matched = []
        for d in docs:
            stem = Path(d.original_filename).stem.lower()
            if stem and stem in q:
                stem_matched.append(d)
        if stem_matched:
            uniq = {d.doc_id: d for d in stem_matched}
            return list(uniq.values()), None

        # Ambiguous query can be resolved by explicit dropdown selection.
        if selected_doc_id:
            rec = self._find_by_id(selected_doc_id)
            if rec is None:
                return [], "Selected file is not available."
            return [rec], None

        if len(docs) == 1:
            return docs, None
        return [], "I found multiple indexed files. Which file would you like to ask about?"

    async def load_rag_for_existing_index(self, rec: DocumentRecord) -> RAGAnything:
        if not _is_index_ready(rec.working_dir):
            raise RuntimeError(
                f"Index for {rec.original_filename} is missing or invalid. "
                "Please reprocess this file."
            )
        if rec.doc_id not in self.rag_cache:
            parser_for_load = (
                rec.parser
                if rec.parser in SUPPORTED_QUERY_LOAD_PARSERS
                else "docling"
            )
            logger.info(
                "Query-only mode: loading existing index for %s from %s",
                rec.original_filename,
                rec.working_dir,
            )
            with self._patched_env_for_ingest():
                rag = await self._create_rag(
                    working_dir=rec.working_dir,
                    parser=parser_for_load,
                )
                init_result = await rag._ensure_lightrag_initialized()
            if not init_result or not init_result.get("success"):
                detail = (init_result or {}).get("error", "unknown error")
                raise RuntimeError(
                    f"failed to initialize existing LightRAG index for "
                    f"{rec.original_filename}: {detail}"
                )
            self.rag_cache[rec.doc_id] = rag
        return self.rag_cache[rec.doc_id]

    async def query_existing_document(
        self,
        question: str,
        rec: DocumentRecord,
        use_direct_vlm_on_query: bool = False,
    ) -> str:
        rag = await self.load_rag_for_existing_index(rec)

        async def _run_query(vlm_enabled: bool):
            try:
                return await rag.aquery(
                    question,
                    mode="hybrid",
                    vlm_enhanced=vlm_enabled,
                )
            except TypeError as type_exc:
                # Backward-compatible path for test doubles or older wrappers
                # that don't accept `vlm_enhanced` yet.
                if "vlm_enhanced" not in str(type_exc):
                    raise
                return await rag.aquery(question, mode="hybrid")

        try:
            result = await _run_query(bool(use_direct_vlm_on_query))
            answer = str(result) if result is not None else "No answer was returned."
            return f"Source: {rec.original_filename}\n\n{answer}"
        except Exception as exc:
            if not use_direct_vlm_on_query:
                raise
            if _is_temporary_vlm_error(exc):
                logger.warning(
                    "VLM query failed with temporary API error; falling back to indexed text/visual descriptions."
                )
                try:
                    result = await _run_query(False)
                    answer = (
                        str(result) if result is not None else "No answer was returned."
                    )
                    return (
                        f"Source: {rec.original_filename}\n\n"
                        "Note: direct vision query was unavailable, so this answer is based "
                        "on the indexed image description.\n\n"
                        f"{answer}"
                    )
                except Exception:
                    raise RuntimeError(
                        "Gemini Vision is temporarily unavailable and no indexed visual "
                        "description could be retrieved."
                    ) from exc
            raise

    async def query(
        self,
        question: str,
        selected_doc_id: Optional[str] = None,
        use_direct_vlm_on_query: bool = False,
    ) -> str:
        routed_docs, route_message = self.route_query_to_documents(
            question, selected_doc_id
        )
        if route_message:
            return route_message

        self.state.query_count += 1
        if len(routed_docs) == 1:
            rec = routed_docs[0]
            try:
                source_prefix = "Answer based on selected file" if selected_doc_id else "Answer based on"
                answer = await self.query_existing_document(
                    question,
                    rec,
                    use_direct_vlm_on_query=use_direct_vlm_on_query,
                )
                return answer.replace("Source:", f"{source_prefix}:")
            except Exception as exc:
                if _is_gemini_quota_exceeded(exc):
                    return (
                        "Gemini quota exceeded. Continue with text/table-only mode "
                        "or wait for quota reset."
                    )
                return f"Query failed: {exc}"

        lines = ["Compared files:"]
        segments = []
        for rec in routed_docs:
            lines.append(f"- {rec.original_filename}")
            try:
                result = await self.query_existing_document(
                    question,
                    rec,
                    use_direct_vlm_on_query=use_direct_vlm_on_query,
                )
                segments.append(f"[{rec.original_filename}]\n{result}")
            except Exception as exc:
                segments.append(f"[{rec.original_filename}]\nQuery failed: {exc}")
        return "\n".join(lines) + "\n\n" + "\n\n".join(segments)

    async def analyze_visual_target(
        self,
        doc_id: str,
        vision_target: Optional[str],
        vision_page_range: Optional[str],
        max_vision_pages: int,
    ) -> str:
        rec = self._find_by_id(doc_id)
        if rec is None:
            return "Please select an indexed file first."
        if rec.file_type != ".pdf":
            return "Visual target analysis is intended for PDF files."

        result = await self._process_single_document(
            file_path=rec.stored_file_path,
            force_parser="pdf_hybrid",
            pdf_page_range=None,
            vision_target=vision_target,
            vision_page_range=vision_page_range,
            max_vision_pages=max_vision_pages,
            no_vision=False,
            skip_kg_extraction=True,
        )
        rec2 = self._find_by_id(doc_id)
        if rec2 is not None:
            rec2.visual_targets.append(
                {
                    "vision_target": vision_target or "",
                    "vision_page_range": vision_page_range or "",
                    "max_vision_pages": int(max_vision_pages),
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            self.registry_store.save(self.registry)
        return result

    def delete_document(self, doc_id: str) -> str:
        rec = self._find_by_id(doc_id)
        if rec is None:
            return "Please select an indexed file first."
        self.rag_cache.pop(doc_id, None)
        wd = Path(rec.working_dir)
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        self.registry = [r for r in self.registry if r.doc_id != doc_id]
        self.registry_store.save(self.registry)
        return f"Deleted indexed file: {rec.original_filename}"


def _extract_uploaded_paths(file_input) -> list[str]:
    if not file_input:
        return []
    if isinstance(file_input, (str, Path)):
        return [str(file_input)]
    out = []
    if isinstance(file_input, list):
        for item in file_input:
            if isinstance(item, (str, Path)):
                out.append(str(item))
            elif hasattr(item, "name"):
                out.append(str(item.name))
    elif hasattr(file_input, "name"):
        out.append(str(file_input.name))
    return out


def build_webui():
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is required for WebUI. Install with: "
            ".\\.venv\\Scripts\\python.exe -m pip install gradio"
        ) from exc

    service = WebUIRAGService()

    with gr.Blocks(title="RAG-Anything Local WebUI") as demo:
        gr.Markdown(
            "## RAG-Anything Local WebUI\n"
            "Upload files -> Process/Index once -> One chat for all indexed files."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload files",
                    type="filepath",
                    file_count="multiple",
                    elem_id="upload_files",
                )
                process_btn = gr.Button(
                    "Process / Index", variant="primary", elem_id="process_index_btn"
                )
                reprocess_btn = gr.Button(
                    "Reprocess selected file", elem_id="reprocess_btn"
                )
                delete_btn = gr.Button("Delete selected file", elem_id="delete_btn")
                indexed_files = gr.Dropdown(
                    label="Indexed files", choices=[], elem_id="indexed_files"
                )
                registry_view = gr.Textbox(label="Indexed files list", lines=10)
                process_status = gr.Textbox(label="Status", lines=10)

                with gr.Accordion("Advanced options", open=False):
                    force_parser = gr.Dropdown(
                        choices=[
                            "",
                            "pdf_hybrid",
                            "pdf_fast",
                            "simple_docx",
                            "paddleocr",
                            "docling",
                        ],
                        value="",
                        label="Force parser",
                    )
                    pdf_page_range = gr.Textbox(label="PDF page range")
                    no_vision = gr.Checkbox(
                        value=True, label="No vision (default for PDF)"
                    )
                    skip_kg_extraction = gr.Checkbox(
                        value=True, label="Skip KG extraction"
                    )
                    vision_target = gr.Textbox(label='Vision target (e.g. "Fig. 2")')
                    vision_page_range = gr.Textbox(label="Vision page range")
                    max_vision_pages = gr.Number(
                        value=1, precision=0, label="Max vision pages"
                    )
                    use_direct_vlm_on_query = gr.Checkbox(
                        value=False,
                        label="Use direct VLM on query",
                        elem_id="use_direct_vlm_on_query",
                    )

                gr.Markdown("### Visual analysis on demand")
                visual_doc = gr.Dropdown(
                    label="PDF file for visual analysis",
                    choices=[],
                    elem_id="visual_doc",
                )
                visual_target = gr.Textbox(
                    label='Vision target (e.g. "Fig. 2")', elem_id="visual_target"
                )
                visual_range = gr.Textbox(label="Vision page range")
                analyze_btn = gr.Button("Analyze visual target", elem_id="analyze_btn")

            with gr.Column(scale=2):
                try:
                    chatbot = gr.Chatbot(label="Chat", type="messages", elem_id="chatbot")
                except TypeError:
                    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot")
                question = gr.Textbox(label="Your question", elem_id="chat_question")
                ask_btn = gr.Button("Ask", elem_id="ask_btn")
                clear_btn = gr.Button("Clear chat", elem_id="clear_btn")

        history_state = gr.State([])

        def refresh_doc_choices():
            choices = [f"{r.original_filename} [{r.doc_id}]" for r in service.list_documents()]
            first = choices[0] if choices else None
            return (
                gr.update(choices=choices, value=first),
                gr.update(choices=choices, value=first),
                service.render_registry_text(),
            )

        def _parse_doc_id(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            m = re.search(r"\[([0-9a-f]+)\]$", value)
            return m.group(1) if m else None

        def on_process(
            files,
            f_parser,
            p_range,
            no_v,
            skip_kg,
            v_target,
            v_range,
            max_v_pages,
        ):
            file_paths = _extract_uploaded_paths(files)
            status = service.run(
                service.process_index(
                    file_paths=file_paths,
                    force_parser=(f_parser or None),
                    pdf_page_range=(p_range or None),
                    vision_target=(v_target or None),
                    vision_page_range=(v_range or None),
                    max_vision_pages=int(max_v_pages or 1),
                    no_vision=bool(no_v),
                    skip_kg_extraction=bool(skip_kg),
                )
            )
            doc_dropdown, visual_dropdown, registry_txt = refresh_doc_choices()
            return status, doc_dropdown, visual_dropdown, registry_txt, [], []

        def on_reprocess(
            selected_doc,
            f_parser,
            p_range,
            no_v,
            skip_kg,
            v_target,
            v_range,
            max_v_pages,
        ):
            doc_id = _parse_doc_id(selected_doc)
            if not doc_id:
                return (
                    "Please select an indexed file first.",
                    gr.update(),
                    gr.update(),
                    service.render_registry_text(),
                )
            rec = service._find_by_id(doc_id)
            if rec is None:
                return (
                    "Selected file is not available.",
                    gr.update(),
                    gr.update(),
                    service.render_registry_text(),
                )
            status = service.run(
                service.process_index(
                    file_paths=[rec.stored_file_path],
                    force_parser=(f_parser or rec.parser),
                    pdf_page_range=(p_range or None),
                    vision_target=(v_target or None),
                    vision_page_range=(v_range or None),
                    max_vision_pages=int(max_v_pages or 1),
                    no_vision=bool(no_v),
                    skip_kg_extraction=bool(skip_kg),
                )
            )
            doc_dropdown, visual_dropdown, registry_txt = refresh_doc_choices()
            return status, doc_dropdown, visual_dropdown, registry_txt

        def on_delete(selected_doc):
            doc_id = _parse_doc_id(selected_doc)
            status = (
                service.delete_document(doc_id)
                if doc_id
                else "Please select an indexed file first."
            )
            doc_dropdown, visual_dropdown, registry_txt = refresh_doc_choices()
            return status, doc_dropdown, visual_dropdown, registry_txt

        def on_ask(user_q, chat_history, selected_doc, use_direct_vlm):
            doc_id = _parse_doc_id(selected_doc)
            answer = service.run(
                service.query(
                    user_q,
                    selected_doc_id=doc_id,
                    use_direct_vlm_on_query=bool(use_direct_vlm),
                )
            )
            out = _append_messages(chat_history, user_q, answer)
            return out, out

        def on_clear():
            return [], []

        def on_analyze(selected_doc, v_target, v_range, max_v_pages):
            doc_id = _parse_doc_id(selected_doc)
            status = service.run(
                service.analyze_visual_target(
                    doc_id=doc_id or "",
                    vision_target=(v_target or None),
                    vision_page_range=(v_range or None),
                    max_vision_pages=int(max_v_pages or 1),
                )
            )
            _, _, registry_txt = refresh_doc_choices()
            return status, registry_txt

        process_btn.click(
            on_process,
            [
                file_input,
                force_parser,
                pdf_page_range,
                no_vision,
                skip_kg_extraction,
                vision_target,
                vision_page_range,
                max_vision_pages,
            ],
            [process_status, indexed_files, visual_doc, registry_view, chatbot, history_state],
        )
        reprocess_btn.click(
            on_reprocess,
            [
                indexed_files,
                force_parser,
                pdf_page_range,
                no_vision,
                skip_kg_extraction,
                vision_target,
                vision_page_range,
                max_vision_pages,
            ],
            [process_status, indexed_files, visual_doc, registry_view],
        )
        delete_btn.click(
            on_delete,
            [indexed_files],
            [process_status, indexed_files, visual_doc, registry_view],
        )
        ask_btn.click(
            on_ask,
            [question, history_state, indexed_files, use_direct_vlm_on_query],
            [chatbot, history_state],
        )
        clear_btn.click(on_clear, [], [chatbot, history_state])
        analyze_btn.click(
            on_analyze,
            [visual_doc, visual_target, visual_range, max_vision_pages],
            [process_status, registry_view],
        )
        demo.load(refresh_doc_choices, [], [indexed_files, visual_doc, registry_view])

    return demo


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _resolve_launch_port(host: str = "127.0.0.1") -> Optional[int]:
    env_port = os.getenv("GRADIO_SERVER_PORT")
    preferred = 7860
    if env_port:
        try:
            preferred = int(env_port)
        except ValueError:
            logger.warning(
                "Invalid GRADIO_SERVER_PORT=%s. Falling back to default 7860.",
                env_port,
            )
            preferred = 7860
    if _is_port_available(host, preferred):
        return preferred
    logger.warning(
        "Preferred port %s is busy. Falling back to an auto-selected free port.",
        preferred,
    )
    # Let Gradio pick a free port automatically.
    return None


if __name__ == "__main__":
    app = build_webui()
    host = "127.0.0.1"
    app.launch(server_name=host, server_port=_resolve_launch_port(host))
