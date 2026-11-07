"""Request-level middleware for the gateway: request-id, structured JSON logs.

Ported from companion-app/app/middleware.py. Nothing here depends on
authentication — the headers go on every response, so a 401 or 500 also
gets a request-id.

Content Security Policy is deliberately left to the host app for now;
sawii-persona's own CSP is set up in api/app.py and should not be clobbered.
If a hardened PWA deployment wants a stricter CSP for /api/*, add a
ResponseHeadersMiddleware here with the policy, but don't double-set what
the main app already sets.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import time

from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("sawii.gateway.http")


_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key"}


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a short request id to every response and every log line."""

    async def dispatch(self, request, call_next):
        incoming = (request.headers.get("x-request-id") or "").strip()[:40]
        request_id = re.sub(r"[^A-Za-z0-9_-]", "", incoming) or secrets.token_urlsafe(8)
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                json.dumps({
                    "event": "request.error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                })
            )
            raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        log.info(
            json.dumps({
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": elapsed_ms,
                "ip": request.client.host if request.client else "",
            })
        )
        return response


def safe_header_dump(headers) -> dict:
    """Dump headers with sensitive values redacted. Useful for debug logs."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        out[name] = "[redacted]" if name.lower() in _SENSITIVE_HEADERS else value
    return out
