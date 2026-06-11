from __future__ import annotations

from backend.app.api.chat import ChatService
from backend.app.api._compat import FastAPI
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.core.config import get_default_paths
from backend.app.core.runtime import get_backend_async_runtime
from backend.app.api.reports import router as reports_router

app = FastAPI(title="RAG-Anything Backend Skeleton")

try:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


@app.get("/health")
def health():
    return {"status": "ok", "service": "backend-skeleton"}


@app.get("/health/runtime")
def health_runtime():
    # Keep runtime diagnostics secret-safe: only embedding/Ollama metadata is
    # returned, never API keys or raw environment dumps.
    service = ChatService(get_default_paths())
    diagnostic = get_backend_async_runtime().run(
        service.get_ollama_runtime_diagnostic()
    )
    return {
        "embedding_provider": service.embedding_provider,
        "configured_embedding_model": service.embedding_cfg.model,
        "resolved_embedding_model": diagnostic.get("resolved_model"),
        "embedding_dim": service.embedding_dim,
        "ollama_host": service.ollama_host,
        "ollama_reachable": diagnostic.get("ollama_reachable", False),
        "available_ollama_models": diagnostic.get("available_models", []),
        "runtime_loop_id": get_backend_async_runtime().loop_id,
    }


app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(reports_router)
