from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_webui_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "webui_gradio.py"
    spec = importlib.util.spec_from_file_location("webui_gradio_docker_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_webui_launch_host_reads_env(monkeypatch):
    module = _load_webui_module()
    monkeypatch.setenv("WEBUI_HOST", "0.0.0.0")
    monkeypatch.setenv("WEBUI_PORT", "7860")
    assert module.get_webui_host() == "0.0.0.0"
    assert module.get_webui_port() == 7860


def test_webui_auth_reads_env(monkeypatch):
    module = _load_webui_module()
    monkeypatch.setenv("WEBUI_AUTH_USER", "admin")
    monkeypatch.setenv("WEBUI_AUTH_PASSWORD", "change-me")
    assert module.get_webui_auth() == ("admin", "change-me")


def test_docker_env_defaults_do_not_use_gemini_embedding():
    text = (Path(__file__).resolve().parents[1] / ".env.docker.example").read_text(
        encoding="utf-8"
    )
    assert "EMBEDDING_PROVIDER=ollama" in text
    assert "EMBEDDING_BINDING=ollama" in text
    assert "EMBEDDING_MODEL=nomic-embed-text" in text
    assert "text-embedding-3-large" not in text


def test_docker_compose_contains_required_services():
    text = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "rag-webui:" in text
    assert "ollama:" in text
    assert "ollama-init:" in text
    assert "7860:7860" in text
    assert "ollama_data:/root/.ollama" in text
    assert "./rag_storage:/app/rag_storage" in text
    assert "./output:/app/output" in text


def test_dockerignore_excludes_secrets_and_storage():
    text = (Path(__file__).resolve().parents[1] / ".dockerignore").read_text(
        encoding="utf-8"
    )
    for expected in [".env", ".env.*", ".venv/", "rag_storage/", "output/", ".tmp/"]:
        assert expected in text
    assert "!/.env.docker.example" in text


def test_dockerfile_contains_pdf_export_dependency():
    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "reportlab" in text
    assert "fonts-dejavu-core" in text
    assert "pymupdf" in text
    assert "pdfplumber" in text
    assert "python-docx" in text


def test_registry_stores_relative_paths(tmp_path):
    module = _load_webui_module()
    registry_path = tmp_path / "rag_storage" / "webui_registry.json"
    store = module.RegistryStore(registry_path)
    rec = module.DocumentRecord(
        doc_id="abc123",
        original_filename="dog.jpg",
        stored_file_rel="webui_uploads/abc123_dog.jpg",
        working_dir_rel="webui_docs/abc123",
        file_type=".jpg",
        parser="paddleocr",
        pdf_mode="auto",
        indexed_at="2026-05-26T00:00:00",
        status="indexed",
        summary="",
        sha256="deadbeef",
        source_metadata={},
        visual_targets=[],
    )
    store.save([rec])
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert raw[0]["stored_file_rel"] == "webui_uploads/abc123_dog.jpg"
    assert raw[0]["working_dir_rel"] == "webui_docs/abc123"
    assert "stored_file_path" not in raw[0]
    assert "working_dir" not in raw[0]


def test_registry_migrates_windows_absolute_paths_or_marks_reprocess(tmp_path):
    module = _load_webui_module()
    registry_path = tmp_path / "rag_storage" / "webui_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = [
        {
            "doc_id": "ok1",
            "original_filename": "dog.jpg",
            "stored_file_path": r"D:\TMA\RAG-Anything\rag_storage\webui_uploads\ok1_dog.jpg",
            "working_dir": r"D:\TMA\RAG-Anything\rag_storage\webui_docs\ok1",
            "file_type": ".jpg",
            "parser": "paddleocr",
            "pdf_mode": "auto",
            "indexed_at": "2026-05-26T00:00:00",
            "status": "indexed",
            "summary": "",
            "sha256": "hash1",
            "source_metadata": {},
            "visual_targets": [],
        },
        {
            "doc_id": "bad1",
            "original_filename": "cat.jpg",
            "stored_file_path": r"D:\old\uploads\cat.jpg",
            "working_dir": r"D:\old\docs\cat",
            "file_type": ".jpg",
            "parser": "paddleocr",
            "pdf_mode": "auto",
            "indexed_at": "2026-05-26T00:00:00",
            "status": "indexed",
            "summary": "",
            "sha256": "hash2",
            "source_metadata": {},
            "visual_targets": [],
        },
    ]
    registry_path.write_text(json.dumps(legacy), encoding="utf-8")
    records = module.RegistryStore(registry_path).load()
    good = next(r for r in records if r.doc_id == "ok1")
    bad = next(r for r in records if r.doc_id == "bad1")
    assert good.stored_file_rel == "webui_uploads/ok1_dog.jpg"
    assert good.working_dir_rel == "webui_docs/ok1"
    assert not good.needs_reprocess
    assert bad.needs_reprocess
    assert "could not migrate legacy absolute path" in bad.needs_reprocess_reason


def test_docker_storage_root_resolves_paths(monkeypatch):
    module = _load_webui_module()
    monkeypatch.setattr(
        module, "REGISTRY_PATH", Path("/app/rag_storage/webui_registry.json")
    )
    svc = module.WebUIRAGService.__new__(module.WebUIRAGService)
    resolved = module.WebUIRAGService._resolve_rel(svc, "webui_docs/doc123")
    assert str(resolved).replace("\\", "/").endswith("/app/rag_storage/webui_docs/doc123")


def test_duplicate_hash_with_invalid_index_reprocesses(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)

    service = module.WebUIRAGService()
    img = tmp_path / "dog.jpg"
    img.write_bytes(b"fake-jpg-bytes")
    sha = module._sha256_file(str(img))
    service.registry = [
        module.DocumentRecord(
            doc_id="old123",
            original_filename="dog.jpg",
            stored_file_rel="webui_uploads/old123_dog.jpg",
            working_dir_rel="webui_docs/old123",
            file_type=".jpg",
            parser="paddleocr",
            pdf_mode="auto",
            indexed_at="2026-05-26T00:00:00",
            status="indexed",
            summary="",
            sha256=sha,
            source_metadata={},
            visual_targets=[],
        )
    ]

    from examples import raganything_example as rx

    async def _fake_process_with_rag(*args, **kwargs):
        wd = Path(kwargs["working_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        for name in [
            "graph_chunk_entity_relation.graphml",
            "vdb_chunks.json",
            "kv_store_text_chunks.json",
        ]:
            (wd / name).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(rx, "process_with_rag", _fake_process_with_rag)
    result = service.run(service._process_single_document(str(img)))
    assert "Skipped duplicate:" not in result
    assert result.startswith("Indexed:")


def test_force_reprocess_does_not_skip_duplicate_hash(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    img = tmp_path / "dog.jpg"
    img.write_bytes(b"same-bytes")
    sha = module._sha256_file(str(img))
    valid_wd = docs / "old123"
    valid_wd.mkdir(parents=True, exist_ok=True)
    (valid_wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (valid_wd / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
    service.registry = [
        module.DocumentRecord(
            doc_id="old123",
            original_filename="dog.jpg",
            stored_file_rel="webui_uploads/old123_dog.jpg",
            working_dir_rel="webui_docs/old123",
            file_type=".jpg",
            parser="paddleocr",
            pdf_mode="auto",
            indexed_at="2026-05-26T00:00:00",
            status="indexed",
            summary="",
            sha256=sha,
            source_metadata={},
            visual_targets=[],
        )
    ]
    from examples import raganything_example as rx

    async def _ok(*args, **kwargs):
        wd = Path(kwargs["working_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
        (wd / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rx, "process_with_rag", _ok)
    result = service.run(service._process_single_document(str(img), force_reprocess=True))
    assert "Skipped duplicate:" not in result
    assert result.startswith("Indexed:")


def test_query_uses_resolved_working_dir(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    docs = rag_root / "webui_docs" / "doc123"
    docs.mkdir(parents=True, exist_ok=True)
    for name in [
        "graph_chunk_entity_relation.graphml",
        "vdb_chunks.json",
        "kv_store_text_chunks.json",
    ]:
        (docs / name).write_text("ok", encoding="utf-8")
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    rec = module.DocumentRecord(
        doc_id="doc123",
        original_filename="dog.jpg",
        stored_file_rel="webui_uploads/doc123_dog.jpg",
        working_dir_rel="webui_docs/doc123",
        file_type=".jpg",
        parser="paddleocr",
        pdf_mode="auto",
        indexed_at="2026-05-26T00:00:00",
        status="indexed",
        summary="",
        sha256="h",
        source_metadata={},
        visual_targets=[],
    )

    captured: dict[str, str] = {}

    class _FakeRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

    async def _fake_create_rag(*, working_dir: str, parser: str):
        captured["working_dir"] = working_dir
        return _FakeRag()

    monkeypatch.setattr(service, "_create_rag", _fake_create_rag)
    monkeypatch.setattr(service, "_resolve_query_parser", lambda _: "paddleocr")
    service.run(service.load_rag_for_existing_index(rec))
    assert captured["working_dir"] == str(docs.resolve())


def test_reprocess_deletes_existing_working_dir_before_index(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(service, "_check_pdf_dependencies", lambda: None)
    monkeypatch.setattr(module, "_sha256_file", lambda _: "sha-rp")
    wd = docs / "docabc"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "stale.txt").write_text("stale", encoding="utf-8")
    service.registry = [
        module.DocumentRecord(
            doc_id="docabc",
            original_filename="a.pdf",
            stored_file_rel="webui_uploads/docabc_a.pdf",
            working_dir_rel="webui_docs/docabc",
            file_type=".pdf",
            parser="pdf_hybrid",
            pdf_mode="hybrid",
            indexed_at="2026-05-26T00:00:00",
            status="indexed",
            summary="",
            sha256="sha-rp",
            source_metadata={},
            visual_targets=[],
        )
    ]
    from examples import raganything_example as rx

    async def _ok(*args, **kwargs):
        wd2 = Path(kwargs["working_dir"])
        assert not (wd2 / "stale.txt").exists()
        (wd2 / "vdb_chunks.json").write_text("{}", encoding="utf-8")
        (wd2 / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rx, "process_with_rag", _ok)
    result = service.run(service._process_single_document(str(f), force_reprocess=True))
    assert result.startswith("Indexed:")


def test_delete_document_removes_registry_and_working_dir(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    uploads.mkdir(parents=True, exist_ok=True)
    wd = docs / "d1"
    wd.mkdir(parents=True, exist_ok=True)
    up = uploads / "d1_a.pdf"
    up.write_text("x", encoding="utf-8")
    service.registry = [
        module.DocumentRecord(
            doc_id="d1",
            original_filename="a.pdf",
            stored_file_rel="webui_uploads/d1_a.pdf",
            working_dir_rel="webui_docs/d1",
            file_type=".pdf",
            parser="pdf_hybrid",
            pdf_mode="hybrid",
            indexed_at="2026-05-26T00:00:00",
            status="indexed",
            summary="",
            sha256="h-del",
            source_metadata={},
            visual_targets=[],
        )
    ]
    msg = service.delete_document("d1")
    assert "Deleted indexed file" in msg
    assert not wd.exists()
    assert not up.exists()
    assert service.registry == []


def test_content_already_exists_error_not_triggered_after_reprocess(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(service, "_check_pdf_dependencies", lambda: None)
    monkeypatch.setattr(module, "_sha256_file", lambda _: "sha-ce")
    service.registry = [
        module.DocumentRecord(
            doc_id="dce",
            original_filename="a.pdf",
            stored_file_rel="webui_uploads/dce_a.pdf",
            working_dir_rel="webui_docs/dce",
            file_type=".pdf",
            parser="pdf_hybrid",
            pdf_mode="hybrid",
            indexed_at="2026-05-26T00:00:00",
            status="failed",
            summary="",
            sha256="sha-ce",
            source_metadata={},
            visual_targets=[],
            needs_reprocess=True,
            needs_reprocess_reason="x",
            error_message="x",
        )
    ]
    from examples import raganything_example as rx

    async def _ok(*args, **kwargs):
        wd = Path(kwargs["working_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
        (wd / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rx, "process_with_rag", _ok)
    result = service.run(service._process_single_document(str(f), force_reprocess=True))
    assert "Content already exists" not in result
    assert result.startswith("Indexed:")


def test_query_parser_fallback_avoids_missing_docling(monkeypatch):
    module = _load_webui_module()
    monkeypatch.setattr(module, "get_parser", lambda _: type("P", (), {"check_installation": lambda self: False})())
    assert module.WebUIRAGService._resolve_query_parser("docling") == "paddleocr"


def test_webui_routes_pdf_to_pdf_hybrid():
    module = _load_webui_module()
    assert module.detect_parser_for_file("a.pdf") == "pdf_hybrid"


def test_webui_routes_docx_to_simple_docx():
    module = _load_webui_module()
    assert module.detect_parser_for_file("a.docx") == "simple_docx"


def test_pdf_dependencies_available_in_docker_requirements():
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "python-docx" in text
    assert "reportlab" in text


def test_failed_index_does_not_mark_document_indexed(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    service.registry = [
        module.DocumentRecord(
            doc_id="d1",
            original_filename="a.pdf",
            stored_file_rel="webui_uploads/a.pdf",
            working_dir_rel="webui_docs/d1",
            file_type=".pdf",
            parser="pdf_hybrid",
            pdf_mode="hybrid",
            indexed_at="2026-05-27T00:00:00",
            status="failed",
            summary="",
            sha256="x",
            source_metadata={},
            visual_targets=[],
            needs_reprocess=True,
            needs_reprocess_reason="index failed",
            error_message="index failed",
        )
    ]
    _, route = service.route_query_to_documents("question")
    assert route == "Please upload and process files first."


def test_registry_records_failed_status_with_error(tmp_path):
    module = _load_webui_module()
    registry_path = tmp_path / "rag_storage" / "webui_registry.json"
    store = module.RegistryStore(registry_path)
    rec = module.DocumentRecord(
        doc_id="f1",
        original_filename="broken.pdf",
        stored_file_rel="webui_uploads/f1_broken.pdf",
        working_dir_rel="webui_docs/f1",
        file_type=".pdf",
        parser="pdf_hybrid",
        pdf_mode="hybrid",
        indexed_at="2026-05-27T00:00:00",
        status="failed",
        summary="",
        sha256="h",
        source_metadata={},
        visual_targets=[],
        needs_reprocess=True,
        needs_reprocess_reason="index path missing",
        error_message="index path missing: /app/rag_storage/webui_docs/f1",
    )
    store.save([rec])
    loaded = store.load()[0]
    assert loaded.status == "failed"
    assert "index path missing" in loaded.error_message


def test_docx_dependency_available_or_clear_error(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    f = tmp_path / "a.docx"
    f.write_bytes(b"not-real-docx")

    from examples import raganything_example as rx

    async def _boom(*args, **kwargs):
        raise RuntimeError("simple_docx requires docling, but docling is not installed in Docker.")

    monkeypatch.setattr(rx, "process_with_rag", _boom)
    monkeypatch.setattr(module, "_sha256_file", lambda _: "h1")
    result = service.run(service._process_single_document(str(f), force_parser="simple_docx"))
    assert "Indexing failed for a.docx:" in result
    assert "docling" in result.lower()


def test_process_failure_surfaces_real_exception(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    monkeypatch.setattr(service, "_check_pdf_dependencies", lambda: None)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    from examples import raganything_example as rx

    async def _boom(*args, **kwargs):
        raise RuntimeError("missing dependency pymupdf in Docker")

    monkeypatch.setattr(rx, "process_with_rag", _boom)
    monkeypatch.setattr(module, "_sha256_file", lambda _: "h2")
    result = service.run(service._process_single_document(str(f), force_parser="pdf_hybrid"))
    assert "missing dependency pymupdf in Docker" in result


def test_process_success_creates_working_dir(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    monkeypatch.setattr(service, "_check_pdf_dependencies", lambda: None)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(module, "_sha256_file", lambda _: "h3")

    from examples import raganything_example as rx

    async def _ok(*args, **kwargs):
        wd = Path(kwargs["working_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
        (wd / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rx, "process_with_rag", _ok)
    result = service.run(service._process_single_document(str(f), force_parser="pdf_hybrid"))
    assert result.startswith("Indexed:")
    assert service.registry[-1].status == "indexed"


def test_process_success_but_missing_working_dir_reports_specific_bug(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    monkeypatch.setattr(service, "_check_pdf_dependencies", lambda: None)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(module, "_sha256_file", lambda _: "h4")

    from examples import raganything_example as rx

    async def _ok_no_artifacts(*args, **kwargs):
        return None

    monkeypatch.setattr(rx, "process_with_rag", _ok_no_artifacts)
    result = service.run(service._process_single_document(str(f), force_parser="pdf_hybrid"))
    assert "process completed but working_dir index artifacts were not created" in result


def test_process_failure_surfaces_quota_error_from_doc_status(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    uploads = rag_root / "webui_uploads"
    docs = rag_root / "webui_docs"
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    monkeypatch.setattr(module, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(module, "DOCS_ROOT", docs)
    service = module.WebUIRAGService()
    f = tmp_path / "a.docx"
    f.write_bytes(b"fake")
    monkeypatch.setattr(module, "_sha256_file", lambda _: "h5")

    from examples import raganything_example as rx

    async def _writes_status_then_returns(*args, **kwargs):
        wd = Path(kwargs["working_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc-1": {
                "file_path": "a.docx",
                "status": "failed",
                "error_msg": "RESOURCE_EXHAUSTED quota exceeded 429",
            }
        }
        (wd / "kv_store_doc_status.json").write_text(json.dumps(payload), encoding="utf-8")
        return None

    monkeypatch.setattr(rx, "process_with_rag", _writes_status_then_returns)
    result = service.run(service._process_single_document(str(f), force_parser="simple_docx"))
    assert "LLM quota exceeded" in result


def test_table_question_retrieves_table_block(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    wd = rag_root / "webui_docs" / "doc1"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (wd / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "a": {
                    "content": (
                        "[PDF Table | page=2 | table=2]\n"
                        "| Model | Scale | Accuracy | ROC |\n"
                        "| --- | --- | --- | --- |\n"
                        "| MLP | 2 | 0.91 | 0.88 |"
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    rec = module.DocumentRecord(
        doc_id="doc1",
        original_filename="paper.pdf",
        stored_file_rel="webui_uploads/doc1_paper.pdf",
        working_dir_rel="webui_docs/doc1",
        file_type=".pdf",
        parser="pdf_hybrid",
        pdf_mode="hybrid",
        indexed_at="2026-05-27T00:00:00",
        status="indexed",
        summary="",
        sha256="s1",
        source_metadata={},
        visual_targets=[],
    )
    class _R:
        async def aquery(self, question, mode="hybrid", vlm_enhanced=False):
            return "References\n[PDF Table | label=Table 1 | page=9]"

    async def _fake_load(_rec):
        return _R()

    monkeypatch.setattr(service, "load_rag_for_existing_index", _fake_load)
    answer = service.run(
        service.query_existing_document(
            "In Table 2, what are the Accuracy and ROC values for MLP at Scale 2? Answer only from the table.",
            rec,
        )
    )
    assert "Accuracy: 0.91; ROC: 0.88" == answer


def test_equation_question_retrieves_equation_block(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    wd = rag_root / "webui_docs" / "doc2"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (wd / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "a": {
                    "content": "[PDF Equation | page=3 | label=Accuracy]\nAccuracy = (TP + TN) / (TP + TN + FP + FN)"
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    rec = module.DocumentRecord(
        doc_id="doc2",
        original_filename="paper.pdf",
        stored_file_rel="webui_uploads/doc2_paper.pdf",
        working_dir_rel="webui_docs/doc2",
        file_type=".pdf",
        parser="pdf_hybrid",
        pdf_mode="hybrid",
        indexed_at="2026-05-27T00:00:00",
        status="indexed",
        summary="",
        sha256="s2",
        source_metadata={},
        visual_targets=[],
    )
    class _R:
        async def aquery(self, question, mode="hybrid", vlm_enhanced=False):
            return "[PDF Equation | label=Accuracy | page=3]"

    async def _fake_load(_rec):
        return _R()

    monkeypatch.setattr(service, "load_rag_for_existing_index", _fake_load)
    answer = service.run(
        service.query_existing_document(
            "What is the Accuracy formula shown in the paper?", rec
        )
    )
    assert "Accuracy = (TP + TN) / (TP + TN + FP + FN)" in answer


def test_equation_answer_only_strips_source_prefix(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    wd = rag_root / "webui_docs" / "doc4"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (wd / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "a": {
                    "content": "[PDF Equation | page=3 | label=Accuracy]\nAccuracy = (TP + TN) / (TP + TN + FP + FN)"
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    rec = module.DocumentRecord(
        doc_id="doc4",
        original_filename="paper.pdf",
        stored_file_rel="webui_uploads/doc4_paper.pdf",
        working_dir_rel="webui_docs/doc4",
        file_type=".pdf",
        parser="pdf_hybrid",
        pdf_mode="hybrid",
        indexed_at="2026-05-27T00:00:00",
        status="indexed",
        summary="",
        sha256="s4",
        source_metadata={},
        visual_targets=[],
    )
    class _R:
        async def aquery(self, question, mode="hybrid", vlm_enhanced=False):
            return "[PDF Equation | label=Accuracy | page=3]"

    async def _fake_load(_rec):
        return _R()

    monkeypatch.setattr(service, "load_rag_for_existing_index", _fake_load)
    answer = service.run(
        service.query_existing_document(
            "What is the Accuracy formula shown in the paper? Answer only with the equation.",
            rec,
        )
    )
    assert answer == "Accuracy = (TP + TN) / (TP + TN + FP + FN)"


def test_figure_question_retrieves_visual_description_block(tmp_path, monkeypatch):
    module = _load_webui_module()
    rag_root = tmp_path / "rag_storage"
    wd = rag_root / "webui_docs" / "doc3"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    (wd / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "a": {
                    "content": "[PDF Visual Description | target=Fig. 2 | page=4 | source=vision]\nFig. 2 compares model accuracy and ROC across scales."
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REGISTRY_PATH", rag_root / "webui_registry.json")
    service = module.WebUIRAGService()
    rec = module.DocumentRecord(
        doc_id="doc3",
        original_filename="paper.pdf",
        stored_file_rel="webui_uploads/doc3_paper.pdf",
        working_dir_rel="webui_docs/doc3",
        file_type=".pdf",
        parser="pdf_hybrid",
        pdf_mode="hybrid",
        indexed_at="2026-05-27T00:00:00",
        status="indexed",
        summary="",
        sha256="s3",
        source_metadata={},
        visual_targets=[],
    )
    class _R:
        async def aquery(self, question, mode="hybrid", vlm_enhanced=False):
            return "[PDF Visual Description | target=Fig. 2 | page=4 | source=vision]"

    async def _fake_load(_rec):
        return _R()

    monkeypatch.setattr(service, "load_rag_for_existing_index", _fake_load)
    answer = service.run(service.query_existing_document("What is Fig 2 about?", rec))
    assert "compares model accuracy and ROC" in answer
