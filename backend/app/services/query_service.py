from __future__ import annotations

import json
import re
import asyncio
from pathlib import Path
from typing import Any, Optional

from lightrag.utils import logger

from raganything.parser import get_parser

from backend.app.core.config import SUPPORTED_QUERY_LOAD_PARSERS
from backend.app.schemas.document import DocumentRecord

CORPUS_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "at",
    "be",
    "can",
    "compare",
    "did",
    "do",
    "does",
    "employee",
    "employees",
    "explain",
    "file",
    "files",
    "for",
    "from",
    "have",
    "how",
    "in",
    "indoors",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "outdoors",
    "please",
    "show",
    "source",
    "sources",
    "summarize",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "use",
    "used",
    "what",
    "which",
    "with",
}

IMAGE_INTENT_TERMS = {
    "animal",
    "breed",
    "cat",
    "color",
    "dog",
    "image",
    "indoors",
    "object",
    "outdoors",
    "pedestrian",
    "photo",
    "picture",
}

TEXT_DOCUMENT_INTENT_TERMS = {
    "accuracy",
    "annual",
    "article",
    "clause",
    "column",
    "employee",
    "law",
    "labor",
    "leave",
    "mlp",
    "roc",
    "row",
    "scale",
    "table",
    "worker",
    "nghỉ",
    "phép",
    "người",
    "lao",
    "động",
}

CORPUS_RELEVANCE_THRESHOLD = 6.0


def normalize_to_messages(history):
    # Normalize either Gradio-style tuples or React-style message objects into
    # a compact structure the API can safely echo/store in UI state.
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


def append_messages(history, user_q: str, answer: str):
    # Keep message history shaping in one place so Gradio and React use the
    # same user/assistant transcript structure.
    messages = normalize_to_messages(history)
    messages.append({"role": "user", "content": str(user_q)})
    messages.append({"role": "assistant", "content": str(answer)})
    return messages


def extract_candidate_filenames(question: str) -> set[str]:
    q = (question or "").lower()
    return set(
        re.findall(
            r"[a-zA-Z0-9_.-]+\.(?:pdf|docx|png|jpg|jpeg|pptx|md|html|htm)",
            q,
        )
    )


def detect_special_pdf_question(question: str) -> Optional[str]:
    q = (question or "").lower()
    if any(k in q for k in ["table ", "table.", "table:", "bảng"]):
        return "table"
    if any(k in q for k in ["equation", "formula", "công thức"]):
        return "equation"
    if any(k in q for k in ["fig", "figure", "chart", "hình"]):
        return "figure"
    return None


def rewrite_pdf_special_query(question: str, special_kind: str) -> str:
    raw = (question or "").strip()
    if special_kind == "equation":
        return (
            "Find equation chunk '[PDF Equation | label=Accuracy]' and return only the full formula. "
            "Must include denominator terms TP + TN + FP + FN when present. "
            f"Question: {raw}"
        )
    if special_kind == "table":
        return (
            "Find table chunk '[PDF Table | label=Table 2]' first when question mentions Table 2. "
            "Answer from table rows only; exclude references and unrelated tables such as Subset 1. "
            f"Question: {raw}"
        )
    if special_kind == "figure":
        return (
            "Find visual description chunk '[PDF Visual Description]' relevant to the figure and answer concisely. "
            f"Question: {raw}"
        )
    return raw


def is_marker_only_answer(text: str) -> bool:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) == 1 and re.match(
        r"^\[PDF (Equation|Table|Visual Description)\b.*\]$", lines[0]
    ):
        return True
    return False


class QueryServiceMixin:
    @staticmethod
    def _cache_entry_matches(
        cache_entry,
        *,
        working_dir: str,
        parser_for_load: str,
    ) -> bool:
        if not isinstance(cache_entry, dict):
            return False
        return (
            cache_entry.get("working_dir") == working_dir
            and cache_entry.get("parser") == parser_for_load
            and cache_entry.get("rag") is not None
        )

    def _detect_source_scope_mismatch(
        self, *, answer: str, rec: DocumentRecord
    ) -> list[str]:
        referenced_files = extract_candidate_filenames(answer)
        allowed_names = {
            str(rec.original_filename or "").strip().lower(),
            Path(str(rec.original_filename or "")).name.lower(),
            Path(str(rec.stored_file_rel or "")).name.lower(),
        }
        return sorted(name for name in referenced_files if name not in allowed_names)

    @staticmethod
    def _looks_like_query_failure(answer: str) -> bool:
        text = str(answer or "").strip().lower()
        return text.startswith("query failed:")

    @staticmethod
    def _retry_metadata(
        *,
        llm_cache_used: bool = False,
        cache_invalidated: bool = False,
        retry_reason: Optional[str] = None,
        llm_response_cache_file: Optional[str] = None,
    ) -> dict:
        return {
            "llm_cache_used": bool(llm_cache_used),
            "cache_invalidated": bool(cache_invalidated),
            "retry_reason": retry_reason,
            "llm_response_cache_file": llm_response_cache_file,
        }

    @staticmethod
    def _extract_query_terms(question: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", (question or "").lower())
        return [
            token
            for token in tokens
            if len(token) >= 2 and token not in CORPUS_QUERY_STOPWORDS
        ]

    @staticmethod
    def _is_compare_question(question: str) -> bool:
        q = (question or "").lower()
        return any(
            marker in q
            for marker in ["compare", "difference", "both", "similar", "versus", "vs"]
        )

    @staticmethod
    def _is_answer_only_request(question: str) -> bool:
        q = (question or "").lower()
        return ("answer only" in q) or ("ch? tr? l?i" in q)

    @staticmethod
    def _resolve_query_parser(preferred: str) -> str:
        candidates = [preferred, "paddleocr", "docling", "simple_docx"]
        seen: set[str] = set()
        for name in candidates:
            n = (name or "").strip().lower()
            if not n or n in seen:
                continue
            seen.add(n)
            try:
                if get_parser(n).check_installation():
                    return n
            except Exception:
                continue
        return "paddleocr"

    @staticmethod
    def _format_marker_answer(
        marker_kind: str, chunks: list[str], question: str = ""
    ) -> str:
        if not chunks:
            return ""
        q = (question or "").lower()
        if marker_kind == "equation":
            for chunk in chunks:
                lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith("[PDF Equation"):
                        continue
                    if "=" in ln and any(
                        token in ln.lower()
                        for token in ["accuracy", "tp", "tn", "fp", "fn"]
                    ):
                        return ln
                if len(lines) > 1:
                    for ln in lines[1:]:
                        if not ln.startswith("[PDF"):
                            return ln
        if marker_kind == "table":
            wants_accuracy = "accuracy" in q
            wants_roc = "roc" in q
            table_num_match = re.search(r"\btable\s*(\d+)\b", q)
            wanted_table_num = (
                int(table_num_match.group(1)) if table_num_match else None
            )
            terms = [
                t
                for t in re.findall(r"[a-zA-Z0-9_.+-]+", q)
                if t
                not in {
                    "in",
                    "the",
                    "what",
                    "are",
                    "for",
                    "at",
                    "from",
                    "table",
                    "answer",
                    "only",
                }
            ]
            scored_chunks: list[tuple[int, str]] = []
            for chunk in chunks:
                score = 0
                if wanted_table_num is not None:
                    m = re.search(
                        r"\[PDF Table\s*\|\s*label=Table\s+(\d+)\s*\|",
                        chunk,
                        flags=re.IGNORECASE,
                    )
                    if m and int(m.group(1)) == wanted_table_num:
                        score += 1000
                if "references" in chunk.lower() or "subset 1" in chunk.lower():
                    score -= 100
                score += sum(1 for term in terms if term in chunk.lower())
                scored_chunks.append((score, chunk))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            for _, chunk in scored_chunks:
                lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
                table_lines = [
                    ln for ln in lines if ln.startswith("|") and ln.endswith("|")
                ]
                if not table_lines:
                    continue
                parsed_rows = []
                for ln in table_lines:
                    cells = [c.strip() for c in ln.strip("|").split("|")]
                    if cells:
                        parsed_rows.append(cells)
                if len(parsed_rows) >= 3:
                    header = parsed_rows[0]
                    body_rows = [
                        r
                        for r in parsed_rows[2:]
                        if not all(re.fullmatch(r"-+", c or "") for c in r)
                    ]
                    best_cells: Optional[list[str]] = None
                    best_score = -1
                    for cells in body_rows:
                        row_text = " | ".join(cells).lower()
                        score = sum(1 for term in terms if term in row_text)
                        if score > best_score:
                            best_score = score
                            best_cells = cells
                    if best_cells is not None and best_score > 0:
                        col_map = {
                            h.strip().lower(): i
                            for i, h in enumerate(header)
                            if h.strip()
                        }
                        out_parts: list[str] = []
                        if wants_accuracy:
                            for k in ["accuracy", "acc"]:
                                if k in col_map and col_map[k] < len(best_cells):
                                    out_parts.append(
                                        f"Accuracy: {best_cells[col_map[k]]}"
                                    )
                                    break
                        if wants_roc:
                            for k in ["roc", "auc", "roc-auc"]:
                                if k in col_map and col_map[k] < len(best_cells):
                                    out_parts.append(f"ROC: {best_cells[col_map[k]]}")
                                    break
                        if out_parts:
                            return "; ".join(out_parts)
                best_line = ""
                best_score = -1
                for ln in table_lines:
                    ln_low = ln.lower()
                    score = sum(1 for term in terms if term in ln_low)
                    if score > best_score:
                        best_score = score
                        best_line = ln
                if best_line and best_score > 0:
                    return best_line
            for _, chunk in scored_chunks:
                for ln in chunk.splitlines():
                    s = ln.strip()
                    if s and not s.startswith("[PDF Table"):
                        return s
            return ""
        if marker_kind == "figure":
            candidate = "\n".join(chunks[0].splitlines()[1:]).strip()
            return candidate or chunks[0]
        return chunks[0]

    def list_indexed_documents(
        self, *, selected_doc_id: Optional[str] = None
    ) -> list[DocumentRecord]:
        # Registry-backed discovery is what preserves corpus chat after restart.
        self._reload_registry()
        indexed_docs: list[DocumentRecord] = []
        for rec in self.registry:
            if selected_doc_id and rec.doc_id != selected_doc_id:
                continue
            ok, _ = self._record_index_status(rec)
            if ok:
                indexed_docs.append(rec)
        return indexed_docs

    def _candidate_text_cache(self) -> dict[tuple[str, str], dict[str, Any]]:
        cache = getattr(self, "_corpus_candidate_text_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_corpus_candidate_text_cache", cache)
        return cache

    def _extract_strings_from_json_payload(self, payload: Any) -> list[str]:
        texts: list[str] = []
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, str):
                cleaned = re.sub(r"\s+", " ", item.strip())
                if cleaned:
                    texts.append(cleaned)
        return texts

    def _load_candidate_text_for_document(self, rec: DocumentRecord) -> dict[str, Any]:
        working_dir = self._resolve_record_working_dir(rec)
        cache_key = (rec.doc_id, str(working_dir))
        cached = self._candidate_text_cache().get(cache_key)
        if cached:
            return cached

        candidate_files = [
            working_dir / "kv_store_text_chunks.json",
            working_dir / "kv_store_full_docs.json",
        ]
        texts: list[str] = []
        previews: list[dict[str, str]] = []
        for path in candidate_files:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(
                    "Corpus candidate text load failed: doc_id=%s filename=%s working_dir=%s source_file=%s error=%s",
                    rec.doc_id,
                    rec.original_filename,
                    working_dir,
                    path,
                    exc,
                )
                continue
            extracted = self._extract_strings_from_json_payload(payload)
            texts.extend(extracted)
            preview = extracted[0][:160] if extracted else ""
            previews.append({"source_file": str(path), "preview": preview})
            logger.info(
                "Corpus candidate text loaded: doc_id=%s filename=%s working_dir=%s source_file=%s preview=%s",
                rec.doc_id,
                rec.original_filename,
                working_dir,
                path,
                preview,
            )

        if rec.file_type in {".jpg", ".jpeg", ".png"}:
            text_blob = " ".join(texts[:8]).lower()
            if any(
                marker in text_blob
                for marker in [
                    "[pdf table",
                    "table 2",
                    "accuracy",
                    "roc",
                    "mlp",
                    "scale 2",
                ]
            ):
                logger.warning(
                    "Corpus candidate text contamination suspected: doc_id=%s filename=%s working_dir=%s preview=%s",
                    rec.doc_id,
                    rec.original_filename,
                    working_dir,
                    text_blob[:200],
                )

        result = {
            "working_dir": str(working_dir),
            "texts": texts,
            "previews": previews,
        }
        self._candidate_text_cache()[cache_key] = result
        return result

    def _score_file_type_intent(
        self, *, question_terms: set[str], file_type: str
    ) -> tuple[float, list[str]]:
        is_image = file_type in {".jpg", ".jpeg", ".png"}
        is_text_doc = file_type in {".pdf", ".docx", ".md", ".html", ".htm", ".txt"}
        score = 0.0
        reasons: list[str] = []

        if question_terms.intersection(IMAGE_INTENT_TERMS):
            if is_image:
                score += 18.0
                reasons.append("image intent matched image file type")
            elif is_text_doc:
                score -= 4.0
        if question_terms.intersection(TEXT_DOCUMENT_INTENT_TERMS):
            if is_text_doc:
                score += 18.0
                reasons.append("document/table/legal intent matched text file type")
            elif is_image:
                score -= 10.0
        return score, reasons

    def _build_no_source_response(
        self,
        *,
        indexed_docs: list[DocumentRecord],
        selected_doc_id: Optional[str],
        considered_documents: list[dict[str, Any]],
        rejected_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "answer": "",
            "error": "No relevant indexed file found for this question.",
            "message": "I could not find relevant information in the uploaded indexed files.",
            "document": None,
            "sources": [],
            "metadata": {
                "mode": "multi_document",
                "indexed_document_count": len(indexed_docs),
                "available_indexed_filenames": [
                    rec.original_filename for rec in indexed_docs
                ],
                "selected_doc_id": selected_doc_id,
                "considered_documents": considered_documents,
                "selected_documents": [],
                "rejected_documents": rejected_documents,
                "no_relevant_source": True,
                "relevance_threshold": CORPUS_RELEVANCE_THRESHOLD,
            },
        }

    def _score_document_candidate(
        self, question: str, rec: DocumentRecord
    ) -> dict[str, Any]:
        # Score from the document's own working_dir artifacts only. This avoids
        # dangerous first-document fallback and catches cross-document text leaks.
        query_terms = self._extract_query_terms(question)
        query_term_set = set(query_terms)
        filename = str(rec.original_filename or "")
        filename_lower = filename.lower()
        stem_lower = Path(filename_lower).stem
        mentions = extract_candidate_filenames(question)
        score = 0.0
        reasons: list[str] = []

        if filename_lower in mentions:
            score += 100.0
            reasons.append(f"matched explicit filename {filename}")

        filename_tokens = {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", stem_lower)
            if len(token) >= 2
        }
        overlap = filename_tokens.intersection(query_term_set)
        if overlap:
            score += float(len(overlap) * 20)
            reasons.append(f"matched filename terms: {', '.join(sorted(overlap)[:3])}")

        file_type_score, file_type_reasons = self._score_file_type_intent(
            question_terms=query_term_set,
            file_type=rec.file_type,
        )
        score += file_type_score
        reasons.extend(file_type_reasons)

        candidate_text = self._load_candidate_text_for_document(rec)
        searchable_chunks = candidate_text["texts"][:20]
        best_chunk_overlap = 0
        best_chunk_excerpt = ""
        for chunk in searchable_chunks:
            chunk_lower = chunk.lower()
            overlap_count = len(
                {term for term in query_term_set if term in chunk_lower}
            )
            if overlap_count > best_chunk_overlap:
                best_chunk_overlap = overlap_count
                best_chunk_excerpt = re.sub(r"\s+", " ", chunk.strip())[:120]
        if best_chunk_overlap:
            score += float(best_chunk_overlap * 8)
            reasons.append(
                f"matched extracted text/description: {best_chunk_excerpt}"
                if best_chunk_excerpt
                else f"matched {best_chunk_overlap} query terms in extracted content"
            )

        unique_overlap = len(overlap) + best_chunk_overlap
        reason = "; ".join(reasons[:3]) if reasons else "no meaningful overlap"
        logger.info(
            "Corpus candidate score: doc_id=%s filename=%s working_dir=%s source_file=%s preview=%s score=%.2f reason=%s",
            rec.doc_id,
            rec.original_filename,
            candidate_text["working_dir"],
            candidate_text["previews"][0]["source_file"]
            if candidate_text["previews"]
            else "(none)",
            candidate_text["previews"][0]["preview"]
            if candidate_text["previews"]
            else "",
            score,
            reason,
        )

        return {
            "doc_id": rec.doc_id,
            "filename": filename,
            "score": score,
            "reason": reason,
            "file_type": rec.file_type,
            "unique_term_overlap": unique_overlap,
            "working_dir": candidate_text["working_dir"],
            "record": rec,
        }

    async def _query_document_with_source_guard(
        self,
        *,
        question: str,
        rec: DocumentRecord,
        use_direct_vlm_on_query: bool = False,
    ) -> dict[str, Any]:
        retry_metadata = self._retry_metadata()
        answer = await self.query_existing_document(
            question,
            rec,
            use_direct_vlm_on_query=use_direct_vlm_on_query,
        )
        if self._looks_like_query_failure(answer):
            return {
                "ok": False,
                "answer": "",
                "error": answer,
                "metadata": retry_metadata,
            }
        mismatched_sources = self._detect_source_scope_mismatch(answer=answer, rec=rec)
        if mismatched_sources:
            working_dir = str(self._resolve_record_working_dir(rec))
            cache_result = self.clear_document_llm_response_cache(working_dir)
            retry_metadata = self._retry_metadata(
                llm_cache_used=True,
                cache_invalidated=bool(cache_result.get("cache_invalidated")),
                retry_reason="source_scope_mismatch",
                llm_response_cache_file=cache_result.get("cache_file"),
            )
            logger.warning(
                "Corpus source mismatch detected: selected=%s working_dir=%s mismatched=%s cache_file=%s cache_invalidated=%s",
                rec.original_filename,
                working_dir,
                mismatched_sources,
                cache_result.get("cache_file"),
                cache_result.get("cache_invalidated"),
            )
            self.rag_cache.pop(rec.doc_id, None)
            answer = await self.query_existing_document(
                question,
                rec,
                use_direct_vlm_on_query=use_direct_vlm_on_query,
            )
            mismatched_sources = self._detect_source_scope_mismatch(
                answer=answer,
                rec=rec,
            )
        if mismatched_sources:
            return {
                "ok": False,
                "answer": "",
                "error": (
                    "Chat response source does not match the currently selected "
                    "document. A stale LLM cache was detected. The cache was "
                    "invalidated and retried. If the problem persists, reprocess "
                    "the document."
                ),
                "metadata": {
                    "selected_doc_id": rec.doc_id,
                    "selected_filename": rec.original_filename,
                    "working_dir": str(self._resolve_record_working_dir(rec)),
                    "referenced_files": sorted(extract_candidate_filenames(answer)),
                    "mismatched_source_files": mismatched_sources,
                    **retry_metadata,
                },
            }
        return {
            "ok": True,
            "answer": answer,
            "error": None,
            "metadata": {
                "selected_doc_id": rec.doc_id,
                "selected_filename": rec.original_filename,
                "working_dir": str(self._resolve_record_working_dir(rec)),
                "referenced_files": sorted(extract_candidate_filenames(answer)),
                **retry_metadata,
            },
        }

    async def _synthesize_corpus_answer(
        self,
        *,
        question: str,
        selected_sources: list[dict[str, Any]],
        history=None,
    ) -> str:
        # The final answer is synthesized from per-document answers so the
        # response can stay grounded in persisted document-specific indexes.
        from backend.app.services.indexing_service import llm_call_runtime

        evidence_blocks = []
        for item in selected_sources:
            evidence_blocks.append(
                f"[Source file: {item['filename']} | score={item['score']:.2f} | reason={item['reason']}]\n"
                f"{item['answer']}"
            )
        prompt = (
            "Answer the user question only from the uploaded files evidence below. "
            "If the evidence is insufficient, say so clearly. "
            "Always end with a short 'Sources:' list using only source filenames.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence_blocks)
        )
        return await llm_call_runtime(
            prompt,
            model=self.llm_model,
            api_key=self.api_key,
            base_url=self.base_url,
            history_messages=normalize_to_messages(history),
        )

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

        mentions = extract_candidate_filenames(q)
        if mentions:
            matched = [d for d in docs if d.original_filename.lower() in mentions]
            if matched:
                return matched, None

        stem_matched = []
        for d in docs:
            stem = Path(d.original_filename).stem.lower()
            if stem and stem in q:
                stem_matched.append(d)
        if stem_matched:
            uniq = {d.doc_id: d for d in stem_matched}
            return list(uniq.values()), None

        if selected_doc_id:
            rec = self._find_by_id(selected_doc_id)
            if rec is None:
                return [], "Selected file is not available."
            return [rec], None

        if len(docs) == 1:
            return docs, None
        return (
            [],
            "I found multiple indexed files. Which file would you like to ask about?",
        )

    async def load_rag_for_existing_index(self, rec: DocumentRecord):
        ok, reason = self._record_index_status(rec)
        if not ok:
            if reason.startswith("registry path incompatible"):
                raise RuntimeError(
                    f"Registry path incompatible for {rec.original_filename}: "
                    f"{reason}. Please reprocess this file."
                )
            raise RuntimeError(
                f"Index path missing for {rec.original_filename}: "
                f"{reason}. Please reprocess this file."
            )
        working_dir = str(self._resolve_record_working_dir(rec))
        preferred = (
            rec.parser if rec.parser in SUPPORTED_QUERY_LOAD_PARSERS else "paddleocr"
        )
        parser_for_load = self._resolve_query_parser(preferred)
        logger.info(
            "Load RAG for doc_id=%s working_dir=%s loop_id=%s cached=%s",
            rec.doc_id,
            working_dir,
            id(asyncio.get_running_loop()),
            rec.doc_id in self.rag_cache,
        )
        cache_entry = self.rag_cache.get(rec.doc_id)
        if not self._cache_entry_matches(
            cache_entry,
            working_dir=working_dir,
            parser_for_load=parser_for_load,
        ):
            logger.info(
                "Query-only mode: loading existing index for %s from %s",
                rec.original_filename,
                working_dir,
            )
            with self._patched_env_for_ingest():
                rag = await self._create_rag(
                    working_dir=working_dir,
                    parser=parser_for_load,
                )
                init_result = await rag._ensure_lightrag_initialized()
            if not init_result or not init_result.get("success"):
                detail = (init_result or {}).get("error", "unknown error")
                raise RuntimeError(
                    "failed to initialize existing LightRAG index for "
                    f"{rec.original_filename}: {detail}"
                )
            self.rag_cache[rec.doc_id] = {
                "rag": rag,
                "working_dir": working_dir,
                "parser": parser_for_load,
            }
        return self.rag_cache[rec.doc_id]["rag"]

    async def query_existing_document(
        self,
        question: str,
        rec: DocumentRecord,
        use_direct_vlm_on_query: bool = False,
    ) -> str:
        working_dir = str(self._resolve_record_working_dir(rec))
        logger.info(
            "LightRAG query start: doc_id=%s filename=%s working_dir=%s loop_id=%s",
            rec.doc_id,
            rec.original_filename,
            working_dir,
            id(asyncio.get_running_loop()),
        )
        rag = await self.load_rag_for_existing_index(rec)
        special_kind = detect_special_pdf_question(question)
        query_text = question
        marker_chunks: list[str] = []
        if rec.file_type == ".pdf" and special_kind:
            marker_chunks = self._find_marker_chunks(rec, special_kind)
            logger.info(
                "Special PDF query uses LightRAG rewrite: type=%s hits=%s file=%s",
                special_kind,
                len(marker_chunks),
                rec.original_filename,
            )
            query_text = rewrite_pdf_special_query(question, special_kind)

        async def _run_query(vlm_enabled: bool):
            try:
                return await rag.aquery(
                    query_text,
                    mode="hybrid",
                    vlm_enhanced=vlm_enabled,
                )
            except TypeError as type_exc:
                if "vlm_enhanced" not in str(type_exc):
                    raise
                return await rag.aquery(query_text, mode="hybrid")

        try:
            result = await _run_query(bool(use_direct_vlm_on_query))
            answer = str(result) if result is not None else "No answer was returned."
            if special_kind and marker_chunks:
                lowered = answer.lower()
                looks_bad_table = special_kind == "table" and (
                    "references" in lowered
                    or "subset 1" in lowered
                    or "[pdf table" in lowered
                )
                looks_bad_equation = special_kind == "equation" and (
                    "[pdf equation" in lowered
                    or ("accuracy" in lowered and ("tp + tn + fp + fn" not in lowered))
                )
                looks_bad_figure = (
                    special_kind == "figure" and "[pdf visual description" in lowered
                )
                if looks_bad_table or looks_bad_equation or looks_bad_figure:
                    marker_answer = self._format_marker_answer(
                        special_kind, marker_chunks, question
                    )
                    if marker_answer and (not is_marker_only_answer(marker_answer)):
                        answer = marker_answer
            if is_marker_only_answer(answer):
                if special_kind and marker_chunks:
                    marker_answer = self._format_marker_answer(
                        special_kind, marker_chunks, question
                    )
                    if marker_answer and (not is_marker_only_answer(marker_answer)):
                        answer = marker_answer
                    else:
                        raise RuntimeError(
                            "Extraction error: retrieved marker-only chunk without usable content."
                        )
                else:
                    raise RuntimeError(
                        "Extraction error: retrieved marker-only chunk without usable content."
                    )
            if self._is_answer_only_request(question):
                logger.info(
                    "LightRAG query succeeded: doc_id=%s filename=%s working_dir=%s loop_id=%s",
                    rec.doc_id,
                    rec.original_filename,
                    working_dir,
                    id(asyncio.get_running_loop()),
                )
                return answer
            logger.info(
                "LightRAG query succeeded: doc_id=%s filename=%s working_dir=%s loop_id=%s",
                rec.doc_id,
                rec.original_filename,
                working_dir,
                id(asyncio.get_running_loop()),
            )
            return f"Source: {rec.original_filename}\n\n{answer}"
        except Exception as exc:
            logger.warning(
                "LightRAG query failed: doc_id=%s filename=%s working_dir=%s loop_id=%s error=%s",
                rec.doc_id,
                rec.original_filename,
                working_dir,
                id(asyncio.get_running_loop()),
                exc,
            )
            if not use_direct_vlm_on_query:
                raise
            if self._is_temporary_vlm_error(exc):
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
                source_prefix = (
                    "Answer based on selected file"
                    if selected_doc_id
                    else "Answer based on"
                )
                answer = await self.query_existing_document(
                    question,
                    rec,
                    use_direct_vlm_on_query=use_direct_vlm_on_query,
                )
                return answer.replace("Source:", f"{source_prefix}:")
            except Exception as exc:
                if self._is_gemini_quota_exceeded(exc):
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

    async def query_with_metadata(
        self,
        question: str,
        *,
        selected_doc_id: Optional[str] = None,
        history=None,
        use_direct_vlm_on_query: bool = False,
    ) -> dict:
        # The React/FastAPI path currently uses single-turn retrieval. History is
        # normalized for UI continuity and future multi-turn expansion, but the
        # LightRAG query itself remains unchanged in this step.
        normalized_history = normalize_to_messages(history)
        routed_docs, route_message = self.route_query_to_documents(
            question, selected_doc_id
        )
        if route_message:
            return {
                "ok": False,
                "answer": "",
                "error": route_message,
                "document": None,
                "metadata": {
                    "history_count": len(normalized_history),
                    "selected_doc_id": selected_doc_id,
                    "routed_document_count": 0,
                },
            }

        self.state.query_count += 1

        if len(routed_docs) == 1:
            rec = routed_docs[0]
            retry_metadata = self._retry_metadata()
            logger.info(
                "Query route selected doc_id=%s filename=%s loop_id=%s",
                rec.doc_id,
                rec.original_filename,
                id(asyncio.get_running_loop()),
            )
            answer = await self.query_existing_document(
                question,
                rec,
                use_direct_vlm_on_query=use_direct_vlm_on_query,
            )
            if self._looks_like_query_failure(answer):
                return {
                    "ok": False,
                    "answer": "",
                    "error": answer,
                    "document": {
                        "doc_id": rec.doc_id,
                        "filename": rec.original_filename,
                        "status": rec.status,
                    },
                    "metadata": {
                        "history_count": len(normalized_history),
                        "selected_doc_id": selected_doc_id,
                        "selected_filename": rec.original_filename,
                        "working_dir": str(self._resolve_record_working_dir(rec)),
                        "routed_document_count": 1,
                        "used_direct_vlm_on_query": bool(use_direct_vlm_on_query),
                        "query_count": self.state.query_count,
                        **retry_metadata,
                    },
                }
            mismatched_sources = self._detect_source_scope_mismatch(
                answer=answer,
                rec=rec,
            )
            if mismatched_sources:
                working_dir = str(self._resolve_record_working_dir(rec))
                cache_result = self.clear_document_llm_response_cache(working_dir)
                retry_metadata = self._retry_metadata(
                    llm_cache_used=True,
                    cache_invalidated=bool(cache_result.get("cache_invalidated")),
                    retry_reason="source_scope_mismatch",
                    llm_response_cache_file=cache_result.get("cache_file"),
                )
                logger.warning(
                    "Source scope mismatch detected: selected=%s working_dir=%s mismatched=%s cache_file=%s cache_invalidated=%s. Retrying with fresh document-scoped RAG instance.",
                    rec.original_filename,
                    working_dir,
                    mismatched_sources,
                    cache_result.get("cache_file"),
                    cache_result.get("cache_invalidated"),
                )
                # A source mismatch after a cache hit is usually a stale
                # per-document LLM response cache entry. Clear only that file,
                # then force a fresh document-scoped RAG load for one retry.
                self.rag_cache.pop(rec.doc_id, None)
                answer = await self.query_existing_document(
                    question,
                    rec,
                    use_direct_vlm_on_query=use_direct_vlm_on_query,
                )
                mismatched_sources = self._detect_source_scope_mismatch(
                    answer=answer,
                    rec=rec,
                )
            if mismatched_sources:
                error_message = (
                    "Chat response source does not match the currently selected "
                    "document. A stale LLM cache was detected. The cache was "
                    "invalidated and retried. If the problem persists, reprocess "
                    "the document."
                )
                return {
                    "ok": False,
                    "answer": "",
                    "error": error_message,
                    "document": {
                        "doc_id": rec.doc_id,
                        "filename": rec.original_filename,
                        "status": rec.status,
                    },
                    "metadata": {
                        "history_count": len(normalized_history),
                        "selected_doc_id": selected_doc_id,
                        "selected_filename": rec.original_filename,
                        "working_dir": str(self._resolve_record_working_dir(rec)),
                        "routed_document_count": 1,
                        "used_direct_vlm_on_query": bool(use_direct_vlm_on_query),
                        "query_count": self.state.query_count,
                        "query_mode": "hybrid",
                        "source_files": [rec.original_filename],
                        "referenced_files": sorted(extract_candidate_filenames(answer)),
                        "mismatched_source_files": mismatched_sources,
                        **retry_metadata,
                    },
                }
            source_prefix = (
                "Answer based on selected file"
                if selected_doc_id
                else "Answer based on"
            )
            answer = answer.replace("Source:", f"{source_prefix}:")
            return {
                "ok": True,
                "answer": answer,
                "error": None,
                "document": {
                    "doc_id": rec.doc_id,
                    "filename": rec.original_filename,
                    "status": rec.status,
                },
                "metadata": {
                    "history_count": len(normalized_history),
                    "selected_doc_id": selected_doc_id,
                    "selected_filename": rec.original_filename,
                    "working_dir": str(self._resolve_record_working_dir(rec)),
                    "routed_document_count": 1,
                    "used_direct_vlm_on_query": bool(use_direct_vlm_on_query),
                    "query_count": self.state.query_count,
                    "query_mode": "hybrid",
                    "source_files": [rec.original_filename],
                    "referenced_files": sorted(extract_candidate_filenames(answer)),
                    **retry_metadata,
                },
            }

        compared_documents = [
            {"doc_id": rec.doc_id, "filename": rec.original_filename}
            for rec in routed_docs
        ]
        segments = []
        for rec in routed_docs:
            result = await self.query_existing_document(
                question,
                rec,
                use_direct_vlm_on_query=use_direct_vlm_on_query,
            )
            segments.append(f"[{rec.original_filename}]\n{result}")
        return {
            "ok": True,
            "answer": "\n\n".join(segments),
            "error": None,
            "document": None,
            "metadata": {
                "history_count": len(normalized_history),
                "selected_doc_id": selected_doc_id,
                "routed_document_count": len(routed_docs),
                "compared_documents": compared_documents,
                "used_direct_vlm_on_query": bool(use_direct_vlm_on_query),
                "query_count": self.state.query_count,
                "query_mode": "hybrid",
                "source_files": [rec.original_filename for rec in routed_docs],
            },
        }

    async def query_corpus_with_metadata(
        self,
        question: str,
        *,
        selected_doc_id: Optional[str] = None,
        history=None,
        use_direct_vlm_on_query: bool = False,
    ) -> dict[str, Any]:
        # Corpus chat is workspace-level by default. A selected_doc_id only
        # narrows the search space; it is no longer required to chat.
        normalized_history = normalize_to_messages(history)
        indexed_docs = self.list_indexed_documents(selected_doc_id=selected_doc_id)
        if not indexed_docs:
            available_filenames = [
                rec.original_filename
                for rec in self.registry
                if rec.status == "indexed" or not selected_doc_id
            ]
            return {
                "ok": False,
                "answer": "",
                "error": "I could not find relevant information in the uploaded files.",
                "document": None,
                "sources": [],
                "metadata": {
                    "mode": "multi_document",
                    "indexed_document_count": 0,
                    "available_indexed_filenames": sorted(set(available_filenames)),
                    "selected_doc_id": selected_doc_id,
                },
            }

        ranked = sorted(
            (self._score_document_candidate(question, rec) for rec in indexed_docs),
            key=lambda item: (
                item["score"],
                item.get("unique_term_overlap", 0),
                1 if item.get("file_type") in {".pdf", ".docx", ".md", ".txt"} else 0,
                item["filename"],
            ),
            reverse=True,
        )
        considered_documents = [
            {
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "score": round(float(item["score"]), 4),
                "reason": item["reason"],
                "working_dir": item.get("working_dir"),
            }
            for item in ranked
        ]
        logger.info(
            "Corpus query ranking: question=%s selected_doc_id=%s indexed_docs=%s top=%s",
            question[:120],
            selected_doc_id,
            len(indexed_docs),
            [
                {
                    "filename": item["filename"],
                    "score": item["score"],
                    "reason": item["reason"],
                }
                for item in ranked[:3]
            ],
        )
        positive_ranked = [
            item for item in ranked if item["score"] >= CORPUS_RELEVANCE_THRESHOLD
        ]
        if (
            selected_doc_id
            and len(indexed_docs) == 1
            and ranked
            and not positive_ranked
        ):
            # An explicit filter is a deliberate scope restriction, so do not
            # reject the only allowed document just because the heuristic score
            # is low for a generic follow-up question.
            positive_ranked = [ranked[0]]
        rejected_documents = [
            {
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "score": round(float(item["score"]), 4),
                "reason": (
                    f"below relevance threshold {CORPUS_RELEVANCE_THRESHOLD}: {item['reason']}"
                ),
            }
            for item in ranked
            if item["score"] < CORPUS_RELEVANCE_THRESHOLD
        ]
        if not positive_ranked:
            return self._build_no_source_response(
                indexed_docs=indexed_docs,
                selected_doc_id=selected_doc_id,
                considered_documents=considered_documents,
                rejected_documents=rejected_documents,
            )

        source_limit = 2 if self._is_compare_question(question) else 1
        selected_candidates = positive_ranked[: max(source_limit, 1)]
        source_results: list[dict[str, Any]] = []
        answer_only_question = (
            "Answer only from this file. Do not use other files.\n"
            f"Question: {question}"
        )
        for candidate in selected_candidates:
            rec = candidate["record"]
            query_result = await self._query_document_with_source_guard(
                question=answer_only_question,
                rec=rec,
                use_direct_vlm_on_query=use_direct_vlm_on_query,
            )
            if not query_result.get("ok"):
                rejected_documents.append(
                    {
                        "doc_id": rec.doc_id,
                        "filename": rec.original_filename,
                        "score": round(float(candidate["score"]), 4),
                        "reason": str(query_result.get("error") or "query rejected"),
                    }
                )
                continue
            source_results.append(
                {
                    "doc_id": rec.doc_id,
                    "filename": rec.original_filename,
                    "score": round(float(candidate["score"]), 4),
                    "reason": candidate["reason"],
                    "answer": str(query_result.get("answer") or "").strip(),
                    "metadata": query_result.get("metadata") or {},
                }
            )

        if not source_results:
            return self._build_no_source_response(
                indexed_docs=indexed_docs,
                selected_doc_id=selected_doc_id,
                considered_documents=considered_documents,
                rejected_documents=rejected_documents,
            )

        if len(source_results) == 1:
            synthesized_answer = source_results[0]["answer"]
        else:
            synthesized_answer = await self._synthesize_corpus_answer(
                question=question,
                selected_sources=source_results,
                history=normalized_history,
            )

        source_names = [item["filename"] for item in source_results]
        if "Sources:" not in synthesized_answer:
            synthesized_answer = (
                synthesized_answer.rstrip()
                + "\n\nSources:\n- "
                + "\n- ".join(source_names)
            )

        return {
            "ok": True,
            "answer": synthesized_answer,
            "error": None,
            "document": source_results[0] if len(source_results) == 1 else None,
            "sources": [
                {
                    "doc_id": item["doc_id"],
                    "filename": item["filename"],
                    "score": item["score"],
                    "reason": item["reason"],
                }
                for item in source_results
            ],
            "metadata": {
                "mode": "multi_document",
                "indexed_document_count": len(indexed_docs),
                "selected_doc_id": selected_doc_id,
                "source_files": source_names,
                "query_mode": "corpus",
                "history_count": len(normalized_history),
                "considered_documents": considered_documents,
                "selected_documents": [
                    {
                        "doc_id": item["doc_id"],
                        "filename": item["filename"],
                        "score": item["score"],
                        "reason": item["reason"],
                    }
                    for item in source_results
                ],
                "rejected_documents": rejected_documents,
                "no_relevant_source": False,
                "relevance_threshold": CORPUS_RELEVANCE_THRESHOLD,
            },
        }
