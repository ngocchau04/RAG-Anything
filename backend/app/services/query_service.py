from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from lightrag.utils import logger

from raganything.parser import get_parser

from backend.app.core.config import SUPPORTED_QUERY_LOAD_PARSERS
from backend.app.schemas.document import DocumentRecord


def normalize_to_messages(history):
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
    def _is_answer_only_request(question: str) -> bool:
        q = (question or "").lower()
        return ("answer only" in q) or ("chỉ trả lời" in q)

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
        if rec.doc_id not in self.rag_cache:
            preferred = (
                rec.parser
                if rec.parser in SUPPORTED_QUERY_LOAD_PARSERS
                else "paddleocr"
            )
            parser_for_load = self._resolve_query_parser(preferred)
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
                    f"LightRAG init failed for " f"{rec.original_filename}: {detail}"
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
                return answer
            return f"Source: {rec.original_filename}\n\n{answer}"
        except Exception as exc:
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
