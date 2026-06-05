from __future__ import annotations

from backend.app.api._compat import APIRouter

router = APIRouter()


@router.post("/chat")
def chat(payload: dict | None = None):
    payload = payload or {}
    return {
        "message": "Chat endpoint scaffold created. Gradio remains the active chat UI.",
        "question": payload.get("question"),
        "selected_doc_id": payload.get("selected_doc_id"),
        "status": "placeholder",
    }
