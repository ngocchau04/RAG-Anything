from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Event, Thread
from typing import Awaitable, TypeVar

from lightrag.utils import logger

T = TypeVar("T")


class BackendAsyncRuntime:
    """Run all backend RAG coroutines on one dedicated loop thread.

    LightRAG keeps some async locks in module-level storage. Reusing those locks
    across request-created event loops can trigger "bound to a different event
    loop" failures, so the backend keeps index/query work on one loop.
    """

    def __init__(self) -> None:
        self._ready = Event()
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(
            target=self._run_loop,
            name="backend-rag-runtime",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        logger.info("Backend async runtime started on loop id=%s", id(self._loop))
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro: Awaitable[T]) -> T:
        # Submit every LightRAG coroutine to the same loop so shared async locks
        # stay bound to one runtime context for the whole process.
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def loop_id(self) -> int:
        return id(self._loop)


_runtime: BackendAsyncRuntime | None = None


def get_backend_async_runtime() -> BackendAsyncRuntime:
    global _runtime
    if _runtime is None:
        _runtime = BackendAsyncRuntime()
    return _runtime
