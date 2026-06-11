from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS = 240


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
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
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
    print(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.details}")


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
                f"resolved={runtime.get('resolved_embedding_model')}"
            ),
        )
    )

    status, documents_payload = _request_json("GET", "/documents", base_url=base_url)
    documents = documents_payload.get("documents", []) if status == 200 else []
    indexed_documents = [
        doc
        for doc in documents
        if doc.get("available_for_chat")
        or (doc.get("status") == "indexed" and not bool(doc.get("needs_reprocess")))
    ]
    results.append(
        SmokeResult(
            "documents/indexed",
            bool(indexed_documents),
            (
                f"indexed_for_chat={len(indexed_documents)} files="
                f"{', '.join(doc.get('filename', '?') for doc in indexed_documents[:5])}"
            )
            if indexed_documents
            else "No indexed documents available for corpus chat.",
        )
    )
    if not indexed_documents:
        for result in results:
            _print_result(result)
        return 1

    def run_corpus_case(
        name: str, question: str, *, allow_no_source: bool = False
    ) -> None:
        status_code, payload = _request_json(
            "POST",
            "/chat/corpus",
            base_url=base_url,
            payload={"message": question, "history": []},
        )
        sources = payload.get("sources") or []
        error_text = str(payload.get("error") or payload.get("message") or "").lower()
        no_source_ok = allow_no_source and (
            status_code == 404
            and (
                "no relevant indexed file found" in error_text
                or "could not find relevant information" in error_text
            )
        )
        ok = (
            status_code == 200
            and payload.get("ok") is True
            and bool(str(payload.get("answer", "")).strip())
            and len(sources) > 0
        ) or no_source_ok
        results.append(
            SmokeResult(
                name,
                ok,
                (
                    f"status={status_code} sources={', '.join(source.get('filename', '?') for source in sources)} "
                    f"answer_len={len(str(payload.get('answer', '')))}"
                )
                if status_code == 200 and payload.get("ok") is True
                else f"status={status_code} error={payload.get('error') or payload}",
            )
        )

    first_filename = indexed_documents[0].get("filename", "the first indexed file")
    run_corpus_case(
        "chat/corpus-dog-or-first",
        f"What information can you find in {first_filename}?",
    )
    run_corpus_case(
        "chat/corpus-legal",
        "How many annual leave days does an employee have?",
        allow_no_source=True,
    )
    if any(
        str(doc.get("file_type", "")).lower() in {".pdf", ".docx"}
        for doc in indexed_documents
    ):
        run_corpus_case(
            "chat/corpus-table",
            "In Table 2, what are the Accuracy and ROC values for MLP at Scale 2?",
        )

    for result in results:
        _print_result(result)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
