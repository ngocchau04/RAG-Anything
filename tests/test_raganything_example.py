from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from functools import partial

import pytest


@pytest.fixture(autouse=True)
def _default_embedding_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")


def _load_example_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "raganything_example.py"
    )
    spec = importlib.util.spec_from_file_location("raganything_example", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_webui_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "webui_gradio.py"
    spec = importlib.util.spec_from_file_location("webui_gradio", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_logging_creates_log_dir(monkeypatch, tmp_path):
    module = _load_example_module()
    log_dir = tmp_path / "nested" / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    module.configure_logging()
    assert log_dir.is_dir()
    assert (log_dir / "raganything_example.log").parent == log_dir


@pytest.mark.asyncio
async def test_image_input_does_not_fallback_mineru(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {"process_document_complete_called": False}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list
            captured["file_path"] = file_path

        async def process_document_complete(self, **kwargs):
            captured["process_document_complete_called"] = True

        async def aquery(self, query, mode="hybrid"):
            return "ok"

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "HELLO OCR"
    )
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)
    monkeypatch.setattr(module, "_doc_status_ok", lambda *_: True)

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        max_chars=20000,
        max_paragraphs=100,
        skip_query=True,
    )
    assert captured["process_document_complete_called"] is False
    assert captured["content_list"][0]["type"] == "text"
    assert captured["content_list"][0]["text"] == "HELLO OCR"


@pytest.mark.asyncio
async def test_image_input_with_user_query_does_not_run_demo_queries(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    captured_queries = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

        async def aquery(self, query, mode="hybrid"):
            captured_queries.append(query)
            return "answer"

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "HELLO OCR"
    )
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)
    monkeypatch.setattr(module, "_doc_status_ok", lambda *_: True)

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        cli_queries=["Trong ảnh có gì?"],
        skip_query=False,
    )
    assert captured_queries == ["Trong ảnh có gì?"]


@pytest.mark.asyncio
async def test_pdf_fast_path_does_not_call_docling_or_mineru(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {"process_document_complete_called": False}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list
            captured["file_path"] = file_path

        async def process_document_complete(self, **kwargs):
            captured["process_document_complete_called"] = True

        async def aquery(self, query, mode="hybrid"):
            return "answer"

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *_, **__: [{"type": "text", "text": "PDF FAST TEXT", "page_idx": 0}],
    )

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        skip_query=True,
    )
    assert captured["process_document_complete_called"] is False
    assert captured["content_list"][0]["text"] == "PDF FAST TEXT"


@pytest.mark.asyncio
async def test_pdf_fast_path_forwards_max_pages_and_page_range(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

        async def aquery(self, query, mode="hybrid"):
            return "answer"

    def fake_extract(pdf_path, max_pages=None, page_range=None):
        captured["max_pages"] = max_pages
        captured["page_range"] = page_range
        return [{"type": "text", "text": "chunk", "page_idx": 0}]

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", fake_extract)

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        skip_query=True,
        max_pages=3,
        page_range="1-3",
    )
    assert captured["max_pages"] == 3
    assert captured["page_range"] == "1-3"


@pytest.mark.asyncio
async def test_image_input_ocr_fallback_no_text_returns_clear_message(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    errors = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

        async def aquery(self, query, mode="hybrid"):
            return "answer"

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: ""
    )
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda msg, *args: errors.append(msg % args if args else msg),
    )

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        cli_queries=["Trong ảnh có gì?"],
        skip_query=False,
    )
    assert any(
        "OCR found no text in image; visual description requires a vision model."
        in message
        for message in errors
    )


def test_extract_text_from_image_with_paddleocr(monkeypatch, tmp_path):
    module = _load_example_module()

    class FakeOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image_path, cls=True):
            return [
                [
                    [[0, 0], ("TEXT_MARKER_CPU_ONLY_RAG", 0.99)],
                    [[1, 1], ("TABLE_MARKER_REVENUE_2024", 0.98)],
                ]
            ]

    fake_pkg = types.SimpleNamespace(PaddleOCR=FakeOCR)
    monkeypatch.setitem(__import__("sys").modules, "paddleocr", fake_pkg)
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    text = module._extract_text_from_image_with_paddleocr(str(image_file))
    assert "TEXT_MARKER_CPU_ONLY_RAG" in text
    assert "TABLE_MARKER_REVENUE_2024" in text


def test_main_maps_skip_query_and_query_arguments(monkeypatch):
    module = _load_example_module()
    captured = {}

    async def fake_process_with_rag(*args):
        captured["args"] = args

    monkeypatch.setattr(module, "process_with_rag", fake_process_with_rag)
    monkeypatch.setattr(
        module.asyncio,
        "run",
        lambda coro: __import__("asyncio")
        .get_event_loop_policy()
        .new_event_loop()
        .run_until_complete(coro),
    )

    monkeypatch.setenv("LLM_BINDING_API_KEY", "dummy")
    monkeypatch.setattr(
        "sys.argv",
        [
            "raganything_example.py",
            "inputs/pedestrian.png",
            "--skip-query",
            "--query",
            "Trong ảnh có gì?",
        ],
    )
    module.main()
    assert captured["args"][13] is True
    assert captured["args"][14] == ["Trong ảnh có gì?"]


@pytest.mark.asyncio
async def test_ollama_embedding_provider_does_not_call_gemini(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {}

    class DummyEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            captured["embedding_func"] = func
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    monkeypatch.setattr(module, "EmbeddingFunc", DummyEmbeddingFunc)
    monkeypatch.setattr(module, "RAGAnything", DummyRAG)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "_ensure_ollama_model_available", _noop)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "ocr-text"
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        skip_query=True,
    )
    emb_func = captured["embedding_func"]
    assert isinstance(emb_func, partial)
    assert emb_func.keywords.get("model") == "nomic-embed-text"
    assert emb_func.keywords.get("host") == "http://localhost:11434"
    assert "api_key" not in emb_func.keywords


@pytest.mark.asyncio
async def test_cli_ollama_default_embedding_model_is_local(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {}

    class DummyEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            captured["embedding_dim"] = embedding_dim
            captured["embedding_func"] = func
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "EmbeddingFunc", DummyEmbeddingFunc)
    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_ensure_ollama_model_available", _noop)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "ocr-text"
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        skip_query=True,
    )
    emb_func = captured["embedding_func"]
    assert captured["embedding_dim"] == 768
    assert emb_func.keywords.get("model") == "nomic-embed-local:latest"
    assert emb_func.keywords.get("host") == "http://localhost:11434"


@pytest.mark.asyncio
async def test_ollama_provider_rejects_text_embedding_3_large(monkeypatch, tmp_path):
    module = _load_example_module()
    errors = []
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda msg, *args: errors.append(msg % args if args else msg),
    )
    file_path = tmp_path / "sample.docx"
    file_path.write_text("dummy", encoding="utf-8")
    await module.process_with_rag(
        file_path=str(file_path),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="simple_docx",
        skip_query=True,
    )
    assert any(
        "Embedding provider/model mismatch: Ollama cannot use text-embedding-3-large."
        in m
        for m in errors
    )


@pytest.mark.asyncio
async def test_embedding_provider_model_mismatch_fails_early(monkeypatch, tmp_path):
    module = _load_example_module()
    errors = []

    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda msg, *args: errors.append(msg % args if args else msg),
    )

    file_path = tmp_path / "sample.docx"
    file_path.write_text("dummy", encoding="utf-8")
    await module.process_with_rag(
        file_path=str(file_path),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="simple_docx",
        skip_query=True,
    )
    assert any("Embedding provider/model mismatch" in m for m in errors)


@pytest.mark.asyncio
async def test_embedding_safe_logs_do_not_expose_keys(monkeypatch, tmp_path):
    module = _load_example_module()
    info_logs = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "_ensure_ollama_model_available", _noop)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "ocr-text"
    )
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda msg, *args: info_logs.append(msg % args if args else msg),
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("GEMINI_API_KEY", "SENSITIVE_KEY_SHOULD_NOT_APPEAR")

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        skip_query=True,
    )
    combined = "\n".join(info_logs)
    assert "SENSITIVE_KEY_SHOULD_NOT_APPEAR" not in combined


@pytest.mark.asyncio
async def test_pdf_hybrid_uses_pdf_fast_text_extraction(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {"fast_called": 0}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

        async def process_document_complete(self, **kwargs):
            captured["docling_called"] = True

    def fake_fast(*args, **kwargs):
        captured["fast_called"] += 1
        return [{"type": "text", "text": "PDF FAST TEXT", "page_idx": 0}]

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", fake_fast)
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert captured["fast_called"] == 1


@pytest.mark.asyncio
async def test_pdf_hybrid_extracts_tables_with_pdfplumber_when_available(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module,
        "_extract_tables_from_pdf_pdfplumber",
        lambda *a, **k: [
            {
                "type": "text",
                "text": "[PDF Table | page=1 | table=1]",
                "page_idx": 0,
                "source": "pdfplumber",
            }
        ],
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert any(i.get("source") == "pdfplumber" for i in captured["content_list"])


@pytest.mark.asyncio
async def test_pdf_hybrid_skips_tables_when_pdfplumber_missing(monkeypatch, tmp_path):
    module = _load_example_module()
    infos = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert any("Content insertion duration" in m for m in infos)


@pytest.mark.asyncio
async def test_pdf_hybrid_renders_selected_pages_for_vision(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {"render_called": 0}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

    def fake_render(*a, **k):
        captured["render_called"] += 1
        return [
            {
                "type": "image",
                "img_path": "x.png",
                "page_idx": 2,
                "source": "pdf_page_render",
            }
        ]

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", fake_render)

    async def fake_describe(*args, **kwargs):
        return "visual description"

    monkeypatch.setattr(module, "_describe_image_with_vision", fake_describe)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        page_range="3,5,7-9",
        skip_query=True,
    )
    assert captured["render_called"] == 1
    assert any(i.get("source") == "pdf_vision" for i in captured["content_list"])


@pytest.mark.asyncio
async def test_pdf_hybrid_vision_target_adds_indexed_visual_description(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "Figure 2 details", "page_idx": 3}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(
        module,
        "_render_pdf_pages_for_vision",
        lambda *a, **k: [{"type": "image", "img_path": "x.png", "page_idx": 3}],
    )
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)
    async def fake_describe(*args, **kwargs):
        return "visual summary"

    monkeypatch.setattr(module, "_describe_image_with_vision", fake_describe)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        vision_target="Fig. 2",
        skip_query=True,
    )

    visual_blocks = [
        b for b in captured["content_list"] if b.get("source") == "pdf_vision"
    ]
    assert len(visual_blocks) == 1
    vb = visual_blocks[0]
    assert "[PDF Visual Description | target=Fig. 2 | page=4 | source=vision]" in vb[
        "text"
    ]
    assert vb["page_idx"] == 3
    assert vb["page_num"] == 4
    assert vb["target"] == "Fig. 2"
    assert vb["image_path"] == "x.png"


@pytest.mark.asyncio
async def test_pdf_hybrid_skips_visuals_without_vision_provider(monkeypatch, tmp_path):
    module = _load_example_module()
    infos = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert any(
        "Vision provider not available; skipped PDF page visual descriptions." in m
        for m in infos
    )


@pytest.mark.asyncio
async def test_pdf_hybrid_does_not_call_mineru(monkeypatch, tmp_path):
    module = _load_example_module()
    called = {"parse_complete": 0}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

        async def process_document_complete(self, **kwargs):
            called["parse_complete"] += 1

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert called["parse_complete"] == 0


@pytest.mark.asyncio
async def test_pdf_hybrid_does_not_call_docling_by_default(monkeypatch, tmp_path):
    module = _load_example_module()
    called = {"parse_complete": 0}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

        async def process_document_complete(self, **kwargs):
            called["parse_complete"] += 1

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_query=True,
    )
    assert called["parse_complete"] == 0


def test_main_accepts_vision_target(monkeypatch):
    module = _load_example_module()
    captured = {}

    async def fake_process_with_rag(*args):
        captured["args"] = args

    monkeypatch.setattr(module, "process_with_rag", fake_process_with_rag)
    monkeypatch.setattr(
        module.asyncio,
        "run",
        lambda coro: __import__("asyncio")
        .get_event_loop_policy()
        .new_event_loop()
        .run_until_complete(coro),
    )
    monkeypatch.setenv("LLM_BINDING_API_KEY", "dummy")
    monkeypatch.setattr(
        "sys.argv",
        [
            "raganything_example.py",
            "inputs/a.pdf",
            "--parser",
            "pdf_hybrid",
            "--vision-target",
            "Fig. 2",
            "--skip-query",
        ],
    )
    module.main()
    assert "Fig. 2" in captured["args"]


def test_resolve_vision_target_page_from_text_blocks():
    module = _load_example_module()
    blocks = [
        {"type": "text", "text": "intro page", "page_idx": 0},
        {"type": "text", "text": "As shown in Figure 2: ...", "page_idx": 4},
    ]
    page = module._resolve_vision_target_page_from_text_blocks(blocks, "Fig. 2")
    assert page == 4


@pytest.mark.asyncio
async def test_pdf_hybrid_vision_target_not_found_warns_and_does_not_render(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    warnings = []
    called = {"render": 0}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    def fake_render(*args, **kwargs):
        called["render"] += 1
        return []

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "no figure mention", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", fake_render)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)
    monkeypatch.setattr(
        module.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        vision_target="Fig. 2",
        skip_query=True,
    )
    assert called["render"] == 0
    assert any("Vision target 'Fig. 2' not found in text layer" in w for w in warnings)


@pytest.mark.asyncio
async def test_pdf_hybrid_explicit_vision_page_range_overrides_target(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    infos = []
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    def fake_render(*args, **kwargs):
        captured["page_range"] = kwargs.get("page_range")
        return []

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "Figure 2 here", "page_idx": 2}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", fake_render)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        vision_target="Fig. 2",
        vision_page_range="7-9",
        skip_query=True,
    )
    assert captured["page_range"] == "7-9"
    assert any(
        "Both vision target and vision page range provided; using explicit --vision-page-range"
        in i
        for i in infos
    )


@pytest.mark.asyncio
async def test_pdf_hybrid_vision_target_page_numbering_is_user_facing(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    infos = []
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    def fake_render(*args, **kwargs):
        captured["page_range"] = kwargs.get("page_range")
        return []

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "Figure 2 appears", "page_idx": 3}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", fake_render)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        vision_target="Fig. 2",
        skip_query=True,
    )
    assert captured["page_range"] == "4"
    assert any(
        "Vision target 'Fig. 2' matched PDF page 4 (internal page_idx=3)" in m
        for m in infos
    )


@pytest.mark.asyncio
async def test_pdf_hybrid_logs_cpu_only_hint_when_kg_enabled(monkeypatch, tmp_path):
    module = _load_example_module()
    warnings = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module,
        "_extract_text_from_pdf_fast",
        lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}],
    )
    monkeypatch.setattr(
        module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: []
    )
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)
    monkeypatch.setattr(
        module.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    await module.process_with_rag(
        file_path=str(pdf),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="pdf_fast",
        pdf_mode="hybrid",
        skip_kg_extraction=False,
        skip_query=True,
    )
    assert any(
        "KG extraction is enabled; this may be slow and consume LLM quota." in w
        for w in warnings
    )


def test_import_webui_module():
    module = _load_webui_module()
    assert module is not None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.pdf", "pdf_hybrid"),
        ("a.docx", "simple_docx"),
        ("a.png", "paddleocr"),
        ("a.jpg", "paddleocr"),
        ("a.pptx", "docling"),
        ("a.md", "docling"),
    ],
)
def test_webui_detect_parser_for_file(name, expected):
    module = _load_webui_module()
    assert module.detect_parser_for_file(name) == expected


@pytest.mark.asyncio
async def test_webui_process_index_runs_once_and_query_does_not_reparse(
    monkeypatch, tmp_path
):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "webui_uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "webui_docs")
    service = module.WebUIRAGService()

    called = {"process": 0, "query": 0}

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def aquery(self, q, mode="hybrid"):
            called["query"] += 1
            return "ok"

    async def fake_process_with_rag(**kwargs):
        called["process"] += 1
        return None

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(module.WebUIRAGService, "_create_rag", fake_create_rag)
    monkeypatch.setenv("LLM_BINDING_API_KEY", "dummy")
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    import examples.raganything_example as rx

    monkeypatch.setattr(rx, "process_with_rag", fake_process_with_rag)
    service.api_key = "dummy"

    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4")
    await service.process_index(str(f))
    assert called["process"] == 1

    await service.query("q1")
    await service.query("q2")
    assert called["process"] == 1
    assert called["query"] == 2


def test_webui_default_embedding_is_ollama(monkeypatch):
    module = _load_webui_module()
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    service = module.WebUIRAGService()
    assert service.embedding_provider == "ollama"
    assert service.embedding_model == "nomic-embed-local:latest"


@pytest.mark.asyncio
async def test_webui_and_cli_share_embedding_defaults(monkeypatch, tmp_path):
    example_module = _load_example_module()
    webui_module = _load_webui_module()
    from raganything.config import resolve_embedding_runtime_config

    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    cfg = resolve_embedding_runtime_config(default_provider="openai")
    assert cfg.provider == "ollama"
    assert cfg.model == "nomic-embed-local:latest"
    assert cfg.dim == 768
    assert cfg.ollama_host == "http://localhost:11434"

    service = webui_module.WebUIRAGService()
    assert service.embedding_model == cfg.model
    assert service.embedding_dim == cfg.dim
    assert service.ollama_host == cfg.ollama_host

    captured = {}

    class DummyEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            captured["embedding_dim"] = embedding_dim
            captured["embedding_func"] = func
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(example_module, "EmbeddingFunc", DummyEmbeddingFunc)
    monkeypatch.setattr(example_module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(example_module, "_ensure_ollama_model_available", _noop)
    monkeypatch.setattr(example_module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        example_module, "_extract_text_from_image_with_paddleocr", lambda *_: "ocr-text"
    )

    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image")
    await example_module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        skip_query=True,
    )
    emb_func = captured["embedding_func"]
    assert emb_func.keywords.get("model") == cfg.model
    assert emb_func.keywords.get("host") == cfg.ollama_host


@pytest.mark.asyncio
async def test_cli_image_flow_regression_no_default_embedding_mismatch(
    monkeypatch, tmp_path
):
    module = _load_example_module()
    errors = []

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            return None

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_ensure_ollama_model_available", _noop)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *_, **__: False)
    monkeypatch.setattr(
        module, "_extract_text_from_image_with_paddleocr", lambda *_: "ocr-text"
    )
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda msg, *args: errors.append(msg % args if args else msg),
    )

    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    image_file = tmp_path / "sample.jpg"
    image_file.write_bytes(b"fake-image")
    await module.process_with_rag(
        file_path=str(image_file),
        output_dir=str(tmp_path / "out"),
        api_key="dummy",
        base_url="https://example.com/v1",
        working_dir=str(tmp_path / "wd"),
        parser="paddleocr",
        skip_query=True,
    )
    assert not any("text-embedding-3-large" in m for m in errors)


def test_webui_defaults_to_gemini_31_flash_lite(monkeypatch):
    module = _load_webui_module()
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("FALLBACK_VISION_MODEL", raising=False)
    service = module.WebUIRAGService()
    assert service.llm_model == "gemini-3.1-flash-lite"
    assert service.vision_model == "gemini-3.1-flash-lite"
    assert service.fallback_vision_model == "gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_query_only_does_not_pass_non_pickleable_objects_to_lightrag(
    monkeypatch, tmp_path
):
    module = _load_webui_module()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()

    async def _noop():
        return None

    monkeypatch.setattr(service, "_ensure_ollama_model_available", _noop)
    rag = await service._create_rag(str((tmp_path / "wd").resolve()), "docling")

    assert getattr(rag.llm_model_func, "__self__", None) is None
    assert getattr(rag.vision_model_func, "__self__", None) is None
    assert rag.llm_model_func.func is module._llm_call_runtime
    assert rag.vision_model_func.func is module._vision_call_runtime
    assert rag.embedding_func.func.func is module._ollama_embed_runtime


def test_webui_process_returns_messages_format():
    module = _load_webui_module()
    history = module._append_messages([], "hi", "hello")
    assert isinstance(history, list)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hi"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "hello"


def test_webui_chat_returns_messages_format():
    module = _load_webui_module()
    history = [{"role": "assistant", "content": "ready"}]
    out = module._append_messages(history, "what is fig 3 about?", "answer")
    assert out[-2]["role"] == "user"
    assert out[-1]["role"] == "assistant"


def test_webui_process_does_not_crash_chatbot_messages():
    module = _load_webui_module()
    # Legacy tuple history must be normalized into messages dict format.
    normalized = module._normalize_to_messages([("u1", "a1")])
    assert normalized == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


@pytest.mark.asyncio
async def test_webui_missing_index_message():
    module = _load_webui_module()
    module.REGISTRY_PATH = Path("./rag_storage/_tmp_webui_registry_test_missing.json")
    if module.REGISTRY_PATH.exists():
        module.REGISTRY_PATH.unlink()
    service = module.WebUIRAGService()
    result = await service.query("hello")
    assert result == "Please upload and process files first."


@pytest.mark.asyncio
async def test_webui_image_no_vision_log_is_not_misleading(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "webui_uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "webui_docs")
    service = module.WebUIRAGService()

    async def fake_process_with_rag(**kwargs):
        return None

    async def fake_create_rag(*args, **kwargs):
        class DummyRag:
            async def aquery(self, q, mode="hybrid"):
                return "ok"

        return DummyRag()

    monkeypatch.setattr(module.WebUIRAGService, "_create_rag", fake_create_rag)
    monkeypatch.setenv("LLM_BINDING_API_KEY", "dummy")
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)
    import examples.raganything_example as rx

    monkeypatch.setattr(rx, "process_with_rag", fake_process_with_rag)
    service.api_key = "dummy"

    img = tmp_path / "dog.jpg"
    img.write_bytes(b"fake")
    status = await service.process_index(str(img), no_vision=True)
    assert (
        "no_vision applies to PDF visual page rendering only; direct image input still uses vision."
        in status
    )


def _make_webui_record(module, tmp_path, doc_id, name, sha="x"):
    wd = tmp_path / "docs" / doc_id
    wd.mkdir(parents=True, exist_ok=True)
    return module.DocumentRecord(
        doc_id=doc_id,
        original_filename=name,
        stored_file_path=str((tmp_path / f"{doc_id}_{name}").resolve()),
        working_dir=str(wd.resolve()),
        file_type=Path(name).suffix.lower(),
        parser="pdf_hybrid" if name.endswith(".pdf") else "paddleocr",
        pdf_mode="hybrid",
        indexed_at="2026-01-01T00:00:00",
        status="indexed",
        summary="",
        sha256=sha,
        source_metadata={
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-local:latest",
            "embedding_dim": 768,
        },
        visual_targets=[],
    )


def test_registry_persists_documents(monkeypatch, tmp_path):
    module = _load_webui_module()
    store = module.RegistryStore(tmp_path / "webui_registry.json")
    rec = _make_webui_record(module, tmp_path, "abc123", "a.pdf", sha="hash1")
    store.save([rec])
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].doc_id == "abc123"
    assert loaded[0].original_filename == "a.pdf"


def test_uploaded_file_is_copied_out_of_gradio_temp(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    src = tmp_path / "gradio_temp" / "dog.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"img")
    out = service._persist_upload_file(str(src), "abc123")
    assert str((tmp_path / "uploads").resolve()) in out
    assert Path(out).exists()


@pytest.mark.asyncio
async def test_duplicate_upload_detected_by_hash(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)
    service = module.WebUIRAGService()
    service.api_key = "dummy"
    monkeypatch.setenv("LLM_BINDING_API_KEY", "dummy")

    async def fake_process_with_rag(**kwargs):
        return None

    import examples.raganything_example as rx

    monkeypatch.setattr(rx, "process_with_rag", fake_process_with_rag)
    f = tmp_path / "dup.pdf"
    f.write_bytes(b"same")
    first = await service.process_index([str(f)])
    second = await service.process_index([str(f)])
    assert "Indexed:" in first
    assert "Skipped duplicate:" in second


@pytest.mark.asyncio
async def test_chat_does_not_reprocess_indexed_file(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "paper.pdf", sha="hash-doc1")
    service.registry = [rec]
    process_called = {"n": 0}

    async def fake_process_single(*args, **kwargs):
        process_called["n"] += 1
        return "indexed"

    async def fake_query_existing(question, rec, use_direct_vlm_on_query=False):
        return "Source: paper.pdf\n\nanswer"

    monkeypatch.setattr(service, "_process_single_document", fake_process_single)
    monkeypatch.setattr(service, "query_existing_document", fake_query_existing)
    ans = await service.query("what is in paper.pdf?")
    assert "Answer based on: paper.pdf" in ans
    assert process_called["n"] == 0


def test_route_query_by_filename(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    service.registry = [
        _make_webui_record(module, tmp_path, "d1", "paper.pdf", sha="h1"),
        _make_webui_record(module, tmp_path, "d2", "dog.jpg", sha="h2"),
    ]
    docs, msg = service.route_query_to_documents("What is in dog.jpg?")
    assert msg is None
    assert len(docs) == 1
    assert docs[0].original_filename == "dog.jpg"


def test_route_query_asks_clarification_when_ambiguous(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    service.registry = [
        _make_webui_record(module, tmp_path, "d1", "paper.pdf", sha="h1"),
        _make_webui_record(module, tmp_path, "d2", "dog.jpg", sha="h2"),
    ]
    docs, msg = service.route_query_to_documents("Summarize it")
    assert docs == []
    assert msg == "I found multiple indexed files. Which file would you like to ask about?"


@pytest.mark.asyncio
async def test_answer_includes_source_filename(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "d1", "paper.pdf", sha="h1")
    service.registry = [rec]

    async def fake_query_existing(question, rec, use_direct_vlm_on_query=False):
        return "Source: paper.pdf\n\ncontent"

    monkeypatch.setattr(service, "query_existing_document", fake_query_existing)
    ans = await service.query("What is Table 2 about?")
    assert "Answer based on: paper.pdf" in ans


def test_webui_chatbot_messages_format():
    module = _load_webui_module()
    out = module._append_messages([], "u", "a")
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_vlm_503_falls_back_to_indexed_description(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    calls = []

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def aquery(self, q, mode="hybrid", vlm_enhanced=False):
            calls.append(vlm_enhanced)
            if vlm_enhanced:
                raise RuntimeError(
                    "Error code: 503 - This model is currently experiencing high demand"
                )
            return "indexed visual description answer"

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    out = await service.query_existing_document(
        "what is this picture about?",
        rec,
        use_direct_vlm_on_query=True,
    )
    assert calls == [True, False]
    assert "indexed image description" in out


@pytest.mark.asyncio
async def test_webui_image_query_uses_indexed_description_by_default(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    seen = {"vlm": None}

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def aquery(self, q, mode="hybrid", vlm_enhanced=False):
            seen["vlm"] = vlm_enhanced
            return "answer"

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    out = await service.query("what is this picture about?", selected_doc_id="doc1")
    assert "Answer based on selected file: dog.jpg" in out
    assert seen["vlm"] is False


@pytest.mark.asyncio
async def test_direct_vlm_query_can_be_enabled(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    seen = {"vlm": None}

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def aquery(self, q, mode="hybrid", vlm_enhanced=False):
            seen["vlm"] = vlm_enhanced
            return "answer"

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    out = await service.query(
        "what is this picture about?",
        selected_doc_id="doc1",
        use_direct_vlm_on_query=True,
    )
    assert "Answer based on selected file: dog.jpg" in out
    assert seen["vlm"] is True


@pytest.mark.asyncio
async def test_vlm_failure_message_is_not_index_missing(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def aquery(self, q, mode="hybrid", vlm_enhanced=False):
            raise RuntimeError("Error code: 503 - UNAVAILABLE")

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    out = await service.query(
        "what is this picture about?",
        selected_doc_id="doc1",
        use_direct_vlm_on_query=True,
    )
    assert "Gemini Vision is temporarily unavailable" in out
    assert "missing or invalid" not in out


@pytest.mark.asyncio
async def test_query_only_loads_existing_index_without_in_memory_rag(
    monkeypatch, tmp_path
):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    service.rag_cache = {}
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    called = {"create": 0, "init": 0, "query": 0}

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            called["init"] += 1
            return {"success": True}

        async def aquery(self, q, mode="hybrid"):
            called["query"] += 1
            return "image answer"

    async def fake_create_rag(*args, **kwargs):
        called["create"] += 1
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    ans = await service.query("what is this picture about?", selected_doc_id="doc1")
    assert "Answer based on selected file: dog.jpg" in ans
    assert called["create"] == 1
    assert called["init"] == 1
    assert called["query"] == 1


@pytest.mark.asyncio
async def test_lightrag_init_error_not_reported_as_missing_index(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    rec = _make_webui_record(module, tmp_path, "doc1", "dog.jpg", sha="hash-doc1")
    service.registry = [rec]
    monkeypatch.setattr(module, "_is_index_ready", lambda *_: True)

    class DummyRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": False, "error": "cannot pickle '_asyncio.Future' object"}

    async def fake_create_rag(*args, **kwargs):
        return DummyRag()

    monkeypatch.setattr(service, "_create_rag", fake_create_rag)
    ans = await service.query("what is this picture about?", selected_doc_id="doc1")
    assert "failed to initialize existing LightRAG index for dog.jpg" in ans
    assert "missing or invalid" not in ans


def test_chat_routes_to_selected_file(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    service.registry = [
        _make_webui_record(module, tmp_path, "d1", "dog.jpg", sha="h1"),
        _make_webui_record(module, tmp_path, "d2", "pedestrian.png", sha="h2"),
    ]
    docs, msg = service.route_query_to_documents(
        "what is this picture about?", selected_doc_id="d2"
    )
    assert msg is None
    assert len(docs) == 1
    assert docs[0].doc_id == "d2"


def test_chat_routes_by_filename_overrides_selected(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    service.registry = [
        _make_webui_record(module, tmp_path, "d1", "dog.jpg", sha="h1"),
        _make_webui_record(module, tmp_path, "d2", "pedestrian.png", sha="h2"),
    ]
    docs, msg = service.route_query_to_documents(
        "what is in dog.jpg?", selected_doc_id="d2"
    )
    assert msg is None
    assert len(docs) == 1
    assert docs[0].doc_id == "d1"


def test_chat_asks_clarification_when_ambiguous_without_selection(monkeypatch, tmp_path):
    module = _load_webui_module()
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "DOCS_ROOT", tmp_path / "docs")
    service = module.WebUIRAGService()
    service.registry = [
        _make_webui_record(module, tmp_path, "d1", "dog.jpg", sha="h1"),
        _make_webui_record(module, tmp_path, "d2", "pedestrian.png", sha="h2"),
    ]
    docs, msg = service.route_query_to_documents("what is this picture about?")
    assert docs == []
    assert msg == "I found multiple indexed files. Which file would you like to ask about?"


def test_default_llm_model_is_gemini_31_flash_lite(monkeypatch):
    module = _load_webui_module()
    monkeypatch.delenv("LLM_MODEL", raising=False)
    service = module.WebUIRAGService()
    assert service.llm_model == "gemini-3.1-flash-lite"


def test_default_vision_model_is_gemini_31_flash_lite(monkeypatch):
    module = _load_webui_module()
    monkeypatch.delenv("VISION_MODEL", raising=False)
    service = module.WebUIRAGService()
    assert service.vision_model == "gemini-3.1-flash-lite"


def test_fallback_vision_model_is_gemini_25_flash_lite(monkeypatch):
    module = _load_webui_module()
    monkeypatch.delenv("FALLBACK_VISION_MODEL", raising=False)
    service = module.WebUIRAGService()
    assert service.fallback_vision_model == "gemini-2.5-flash-lite"


def test_env_example_uses_gemini_31_flash_lite():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example is not present in this repository.")
    text = env_example.read_text(encoding="utf-8")
    assert "LLM_MODEL=gemini-3.1-flash-lite" in text
    assert "VISION_MODEL=gemini-3.1-flash-lite" in text
    assert "FALLBACK_VISION_MODEL=gemini-2.5-flash-lite" in text


def test_ollama_embedding_config_unchanged():
    from raganything.config import resolve_embedding_runtime_config

    cfg = resolve_embedding_runtime_config(
        env={
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_BINDING": "ollama",
            "EMBEDDING_MODEL": "nomic-embed-local:latest",
            "EMBEDDING_DIM": "768",
            "OLLAMA_HOST": "http://localhost:11434",
        },
        default_provider="openai",
    )
    assert cfg.provider == "ollama"
    assert cfg.model == "nomic-embed-local:latest"
    assert cfg.dim == 768
    assert cfg.ollama_host == "http://localhost:11434"
