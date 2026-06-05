from __future__ import annotations

from backend.app.api._compat import FastAPI
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
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


app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(reports_router)
