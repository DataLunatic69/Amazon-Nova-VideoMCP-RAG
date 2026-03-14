"""
Middlewares package.

Currently all middleware is registered directly in ``api/app.py``:
  - CORSMiddleware — allows cross-origin requests from any origin.

This package is a hook for adding additional middleware layers, such as:
  - Request-ID injection (attach a unique ID to every request for tracing).
  - Structured access logging (JSON log lines with latency, status, path).
  - Rate limiting (e.g. slowapi + redis for per-IP throttling).
  - API key / Bearer token validation (move from router-level to global).

Example skeleton for a custom middleware:

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class RequestIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            import uuid
            request.state.request_id = uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response

Register it in ``api/app.py``:

    from api.middlewares import RequestIdMiddleware
    application.add_middleware(RequestIdMiddleware)
"""
