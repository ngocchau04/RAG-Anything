from __future__ import annotations

try:
    from fastapi import APIRouter, FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    class _StubRouter:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def add_api_route(self, path, endpoint, methods=None, **kwargs):
            self.routes.append(
                {
                    "path": path,
                    "endpoint": endpoint,
                    "methods": methods or [],
                    "name": kwargs.get("name") or getattr(endpoint, "__name__", "route"),
                }
            )

        def get(self, path, **kwargs):
            def decorator(func):
                self.add_api_route(path, func, methods=["GET"], **kwargs)
                return func

            return decorator

        def post(self, path, **kwargs):
            def decorator(func):
                self.add_api_route(path, func, methods=["POST"], **kwargs)
                return func

            return decorator

        def delete(self, path, **kwargs):
            def decorator(func):
                self.add_api_route(path, func, methods=["DELETE"], **kwargs)
                return func

            return decorator

    class _StubFastAPI(_StubRouter):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.title = kwargs.get("title", "")

        def include_router(self, router, prefix="", tags=None):
            for route in getattr(router, "routes", []):
                copied = dict(route)
                copied["path"] = f"{prefix}{route['path']}"
                self.routes.append(copied)

    def _identity(*args, **kwargs):
        return None

    class _StubJSONResponse(dict):
        def __init__(self, content=None, status_code=200):
            super().__init__(content=content, status_code=status_code)

    APIRouter = _StubRouter
    FastAPI = _StubFastAPI
    File = _identity
    Form = _identity
    UploadFile = object
    JSONResponse = _StubJSONResponse
