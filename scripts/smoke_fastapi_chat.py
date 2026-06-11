from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class SmokeResult:
    name: str
    ok: bool
    details: str


def _request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            return response.status, data
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        return exc.code, data
    except URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def _print_result(result: SmokeResult) -> None:
    prefix = "PASS" if result.ok else "FAIL"
    print(f"[{prefix}] {result.name}: {result.details}")


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    results: list[SmokeResult] = []

    try:
        status, runtime = _request_json("GET", "/health/runtime", base_url=base_url)
    except Exception as exc:
        _print_result(SmokeResult("health/runtime", False, str(exc)))
        return 1

    runtime_ok = (
        status == 200
        and runtime.get("embedding_provider") == "ollama"
        and runtime.get("ollama_reachable") is True
    )
    results.append(
        SmokeResult(
            "health/runtime",
            runtime_ok,
            (
                f"provider={runtime.get('embedding_provider')} "
                f"configured={runtime.get('configured_embedding_model')} "
                f"resolved={runtime.get('resolved_embedding_model')} "
                f"host={runtime.get('ollama_host')} "
                f"loop={runtime.get('runtime_loop_id')}"
            ),
        )
    )

    status, documents_payload = _request_json("GET", "/documents", base_url=base_url)
    documents = documents_payload.get("documents", []) if status == 200 else []
    indexed_documents = [
        doc
        for doc in documents
        if doc.get("status") == "indexed" and not doc.get("needs_reprocess")
    ]
    if not indexed_documents:
        results.append(
            SmokeResult(
                "documents/indexed",
                False,
                "No indexed document found. Upload and index at least one document before running this smoke test.",
            )
        )
        for result in results:
            _print_result(result)
        return 1

    first_doc = indexed_documents[0]
    results.append(
        SmokeResult(
            "documents/indexed",
            True,
            f"Found {len(indexed_documents)} indexed document(s). First doc={first_doc.get('filename')} ({first_doc.get('doc_id')})",
        )
    )

    def run_chat(name: str, doc: dict[str, Any], question: str) -> None:
        status_code, payload = _request_json(
            "POST",
            "/chat",
            base_url=base_url,
            payload={
                "message": question,
                "selected_doc_id": doc.get("doc_id"),
                "selected_document_id": doc.get("doc_id"),
                "history": [],
                "require_selected_document": True,
            },
        )
        ok = (
            status_code == 200
            and payload.get("ok") is True
            and bool(str(payload.get("answer", "")).strip())
        )
        details = (
            f"doc={doc.get('filename')} status={status_code} "
            f"answer_len={len(str(payload.get('answer', '')))}"
        )
        if not ok:
            details = f"{details} error={payload.get('error') or payload.get('details') or payload}"
        results.append(SmokeResult(name, ok, details))

    # Repeated chat on one indexed document is the main regression case for the
    # shared async runtime and LightRAG lock stability.
    run_chat("chat/first", first_doc, "What is this about?")
    run_chat("chat/repeat-same-document", first_doc, "Summarize this again.")

    if len(indexed_documents) > 1:
        run_chat(
            "chat/other-document",
            indexed_documents[1],
            "What is this document about?",
        )
    else:
        results.append(
            SmokeResult(
                "chat/other-document",
                True,
                "Skipped because only one indexed document is available.",
            )
        )

    for result in results:
        _print_result(result)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
