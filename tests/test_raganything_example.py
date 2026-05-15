from __future__ import annotations

import importlib.util
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
async def test_image_input_with_user_query_does_not_run_demo_queries(monkeypatch, tmp_path):
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
async def test_image_input_ocr_fallback_no_text_returns_clear_message(monkeypatch, tmp_path):
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
    monkeypatch.setattr(module, "_extract_text_from_image_with_paddleocr", lambda *_: "")
    monkeypatch.setattr(module.logger, "error", lambda msg, *args: errors.append(msg % args if args else msg))

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
async def test_embedding_provider_model_mismatch_fails_early(monkeypatch, tmp_path):
    module = _load_example_module()
    errors = []

    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setattr(module.logger, "error", lambda msg, *args: errors.append(msg % args if args else msg))

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
    monkeypatch.setattr(module.logger, "info", lambda msg, *args: info_logs.append(msg % args if args else msg))
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
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
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
async def test_pdf_hybrid_extracts_tables_with_pdfplumber_when_available(monkeypatch, tmp_path):
    module = _load_example_module()
    captured = {}

    class DummyRAG:
        def __init__(self, config=None, **kwargs):
            self.config = config

        async def insert_content_list(self, content_list, file_path=None, **kwargs):
            captured["content_list"] = content_list

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(
        module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}]
    )
    monkeypatch.setattr(
        module,
        "_extract_tables_from_pdf_pdfplumber",
        lambda *a, **k: [{"type": "text", "text": "[PDF Table | page=1 | table=1]", "page_idx": 0, "source": "pdfplumber"}],
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
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}])
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)
    monkeypatch.setattr(module.logger, "info", lambda msg, *args: infos.append(msg % args if args else msg))

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
        return [{"type": "image", "img_path": "x.png", "page_idx": 2, "source": "pdf_page_render"}]

    monkeypatch.setattr(module, "RAGAnything", DummyRAG)
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}])
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
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
    assert any(i.get("source") == "pdf_page_vision" for i in captured["content_list"])


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
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}])
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", lambda *a, **k: [])
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: False)
    monkeypatch.setattr(module.logger, "info", lambda msg, *args: infos.append(msg % args if args else msg))

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
    assert any("Vision provider not available; skipped PDF page visual descriptions." in m for m in infos)


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
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}])
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
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
    monkeypatch.setattr(module, "_extract_text_from_pdf_fast", lambda *a, **k: [{"type": "text", "text": "base", "page_idx": 0}])
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
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
async def test_pdf_hybrid_vision_target_not_found_warns_and_does_not_render(monkeypatch, tmp_path):
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
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
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
async def test_pdf_hybrid_explicit_vision_page_range_overrides_target(monkeypatch, tmp_path):
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
    monkeypatch.setattr(module, "_extract_tables_from_pdf_pdfplumber", lambda *a, **k: [])
    monkeypatch.setattr(module, "_render_pdf_pages_for_vision", fake_render)
    monkeypatch.setattr(module, "_has_vision_provider", lambda *a, **k: True)
    monkeypatch.setattr(
        module.logger, "info", lambda msg, *args: infos.append(msg % args if args else msg)
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
