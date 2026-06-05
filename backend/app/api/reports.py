from __future__ import annotations

from backend.app.api._compat import APIRouter

router = APIRouter()


@router.post("/reports/current-answer")
def report_current_answer(payload: dict | None = None):
    payload = payload or {}
    return {
        "message": "Current-answer report endpoint scaffold created.",
        "status": "placeholder",
        "source_file": payload.get("source_file"),
    }


@router.post("/reports/chat-history")
def report_chat_history(payload: dict | None = None):
    payload = payload or {}
    return {
        "message": "Chat-history report endpoint scaffold created.",
        "status": "placeholder",
        "messages_count": len(payload.get("messages") or []),
    }


@router.post("/reports/agent")
def report_agent(payload: dict | None = None):
    payload = payload or {}
    return {
        "message": "PDF report agent endpoint scaffold created.",
        "status": "placeholder",
        "request": payload.get("report_request"),
    }
