#!/usr/bin/env python
"""
Local Gradio WebUI for RAG-Anything (CPU-only friendly).
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from threading import Thread
from typing import Optional

from dotenv import load_dotenv
from lightrag.utils import logger
from raganything.config import resolve_embedding_runtime_config
from raganything.parser import get_parser

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.core.config import AppPaths, get_default_paths
from backend.app.schemas.document import DocumentRecord, UIState
from backend.app.services.document_service import (
    DocumentServiceMixin,
    RegistryStore,
    is_index_ready,
)
from backend.app.services.indexing_service import (
    IndexingServiceMixin,
    detect_parser_for_file,
    index_artifact_status as _index_artifact_status,
    read_doc_status_error as _read_doc_status_error,
    safe_filename as _safe_filename,
    sha256_file as _sha256_file,
)
from backend.app.services.query_service import (
    QueryServiceMixin,
    append_messages as _append_messages,
    detect_special_pdf_question as _detect_special_pdf_question,
    extract_candidate_filenames as _extract_candidate_filenames,
    is_marker_only_answer as _is_marker_only_answer,
    normalize_to_messages as _normalize_to_messages,
    rewrite_pdf_special_query as _rewrite_pdf_special_query,
)
from backend.app.services.report_service import (
    export_chat_history_action,
    export_current_answer_action,
    format_pdf_export_result as _format_pdf_export_result,
    generate_pdf_report_action,
    parse_doc_id_from_label as _parse_doc_id_from_label,
    parse_doc_label as _parse_doc_label,
)

load_dotenv(dotenv_path=".env", override=False)

_DEFAULT_PATHS = get_default_paths()
REGISTRY_PATH = _DEFAULT_PATHS.registry_path
UPLOADS_DIR = _DEFAULT_PATHS.uploads_dir
DOCS_ROOT = _DEFAULT_PATHS.docs_root
REPORTS_ROOT = _DEFAULT_PATHS.reports_root


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


def _is_index_ready(working_dir: str) -> bool:
    return is_index_ready(working_dir)


class WebUIRAGService(DocumentServiceMixin, QueryServiceMixin, IndexingServiceMixin):
    def __init__(self):
        self.state = UIState()
        self.paths = AppPaths(
            registry_path=REGISTRY_PATH.resolve(),
            uploads_dir=UPLOADS_DIR.resolve(),
            docs_root=DOCS_ROOT.resolve(),
            reports_root=REPORTS_ROOT.resolve(),
        )
        self.registry_store = RegistryStore(self.paths.registry_path)
        self.registry: list[DocumentRecord] = self.registry_store.load()
        if self.registry_store._last_load_migrated:
            self.registry_store.save(self.registry)
        self.rag_cache = {}

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

    @staticmethod
    def _is_gemini_quota_exceeded(exc: Exception) -> bool:
        return _is_gemini_quota_exceeded(exc)

    @staticmethod
    def _is_temporary_vlm_error(exc: Exception) -> bool:
        return _is_temporary_vlm_error(exc)

    @staticmethod
    def _is_index_ready(working_dir: str) -> bool:
        return _is_index_ready(working_dir)

    @staticmethod
    def _sha256_file(path: str) -> str:
        return _sha256_file(path)

    def _resolve_rel(self, rel_path: str) -> Path:
        if hasattr(self, "paths"):
            root = self.paths.storage_root
        else:
            root = REGISTRY_PATH.parent.resolve()
        return (root / Path(rel_path)).resolve()

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

                gr.Markdown("### Export PDF")
                export_current_btn = gr.Button(
                    "Export current answer to PDF", elem_id="export_current_pdf_btn"
                )
                export_history_btn = gr.Button(
                    "Export chat history to PDF", elem_id="export_history_pdf_btn"
                )
                export_status = gr.Textbox(label="Export status", lines=3)
                export_pdf_file = gr.File(label="Download PDF report")
                gr.Markdown("### PDF Report Agent")
                report_request = gr.Textbox(label="Report request")
                generate_report_btn = gr.Button(
                    "Generate PDF report", elem_id="generate_report_btn"
                )
                report_status = gr.Textbox(label="PDF report status", lines=3)
                report_file = gr.File(label="Download generated PDF report")

            with gr.Column(scale=2):
                try:
                    chatbot = gr.Chatbot(label="Chat", type="messages", elem_id="chatbot")
                except TypeError:
                    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot")
                question = gr.Textbox(label="Your question", elem_id="chat_question")
                ask_btn = gr.Button("Ask", elem_id="ask_btn")
                clear_btn = gr.Button("Clear chat", elem_id="clear_btn")

        history_state = gr.State([])
        last_question_state = gr.State("")
        last_answer_state = gr.State("")
        last_source_state = gr.State("")

        def refresh_doc_choices():
            choices = [f"{r.original_filename} [{r.doc_id}]" for r in service.list_documents()]
            first = choices[0] if choices else None
            return (
                gr.update(choices=choices, value=first),
                gr.update(choices=choices, value=first),
                service.render_registry_text(),
            )

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
            doc_id = _parse_doc_id_from_label(selected_doc)
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
                    file_paths=[str(service._resolve_record_stored_file(rec))],
                    force_parser=(f_parser or rec.parser),
                    pdf_page_range=(p_range or None),
                    vision_target=(v_target or None),
                    vision_page_range=(v_range or None),
                    max_vision_pages=int(max_v_pages or 1),
                    no_vision=bool(no_v),
                    skip_kg_extraction=bool(skip_kg),
                    force_reprocess=True,
                )
            )
            doc_dropdown, visual_dropdown, registry_txt = refresh_doc_choices()
            return status, doc_dropdown, visual_dropdown, registry_txt

        def on_delete(selected_doc):
            doc_id = _parse_doc_id_from_label(selected_doc)
            status = (
                service.delete_document(doc_id)
                if doc_id
                else "Please select an indexed file first."
            )
            doc_dropdown, visual_dropdown, registry_txt = refresh_doc_choices()
            return status, doc_dropdown, visual_dropdown, registry_txt

        def on_ask(user_q, chat_history, selected_doc, use_direct_vlm):
            doc_id = _parse_doc_id_from_label(selected_doc)
            answer = service.run(
                service.query(
                    user_q,
                    selected_doc_id=doc_id,
                    use_direct_vlm_on_query=bool(use_direct_vlm),
                )
            )
            out = _append_messages(chat_history, user_q, answer)
            return out, out, str(user_q or ""), str(answer or ""), _parse_doc_label(selected_doc) or ""

        def on_clear():
            return [], [], "", "", ""

        def on_analyze(selected_doc, v_target, v_range, max_v_pages):
            doc_id = _parse_doc_id_from_label(selected_doc)
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

        def on_export_current_answer(last_q, last_a, last_source):
            return export_current_answer_action(
                question=last_q,
                answer=last_a,
                source_file=last_source,
                output_dir=REPORTS_ROOT,
            )

        def on_export_chat_history(chat_history, selected_doc, last_source):
            return export_chat_history_action(
                chat_history=chat_history,
                selected_doc_label=selected_doc,
                last_source=last_source,
                output_dir=REPORTS_ROOT,
            )

        def on_generate_pdf_report(req_text, selected_doc):
            return generate_pdf_report_action(
                report_request=req_text,
                selected_doc_label=selected_doc,
                service=service,
                output_dir=REPORTS_ROOT,
            )

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
            [chatbot, history_state, last_question_state, last_answer_state, last_source_state],
        )
        clear_btn.click(
            on_clear,
            [],
            [chatbot, history_state, last_question_state, last_answer_state, last_source_state],
        )
        analyze_btn.click(
            on_analyze,
            [visual_doc, visual_target, visual_range, max_vision_pages],
            [process_status, registry_view],
        )
        export_current_btn.click(
            on_export_current_answer,
            [last_question_state, last_answer_state, last_source_state],
            [export_status, export_pdf_file],
        )
        export_history_btn.click(
            on_export_chat_history,
            [history_state, indexed_files, last_source_state],
            [export_status, export_pdf_file],
        )
        generate_report_btn.click(
            on_generate_pdf_report,
            [report_request, indexed_files],
            [report_status, report_file],
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_webui_host() -> str:
    return os.getenv("WEBUI_HOST", "0.0.0.0")


def get_webui_port() -> int:
    env_port = os.getenv("WEBUI_PORT") or os.getenv("GRADIO_SERVER_PORT")
    preferred = 7860
    if env_port:
        try:
            preferred = int(env_port)
        except ValueError:
            logger.warning(
                "Invalid WEBUI_PORT/GRADIO_SERVER_PORT=%s. Falling back to default 7860.",
                env_port,
            )
            preferred = 7860
    return preferred


def get_webui_auth() -> Optional[tuple[str, str]]:
    user = os.getenv("WEBUI_AUTH_USER", "").strip()
    password = os.getenv("WEBUI_AUTH_PASSWORD", "").strip()
    if user and password:
        return (user, password)
    return None


def get_webui_share() -> bool:
    return _env_bool("GRADIO_SHARE", default=False)


def _resolve_launch_port(host: str, preferred: int) -> Optional[int]:
    if _is_port_available(host, preferred):
        return preferred
    logger.warning(
        "Preferred port %s is busy. Falling back to an auto-selected free port.",
        preferred,
    )
    return None


if __name__ == "__main__":
    app = build_webui()
    host = get_webui_host()
    preferred_port = get_webui_port()
    auth = get_webui_auth()
    share = get_webui_share()
    if auth:
        logger.info("WebUI auth is enabled.")
    launch_kwargs = {
        "server_name": host,
        "server_port": _resolve_launch_port(host, preferred_port),
        "share": share,
    }
    if auth:
        launch_kwargs["auth"] = auth
    app.launch(**launch_kwargs)
