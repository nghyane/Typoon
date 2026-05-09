"""Request-ID middleware.

Stamps every incoming request with a short id, exposes it via
`request.state.request_id`, and echoes it back in the
`X-Request-ID` response header. Logs surface the same id so a
client-side error report can be matched to server logs without
greping by timestamp.

If the client supplied an inbound `X-Request-ID` we trust it (closed
community deploy — no spoof concern); otherwise we mint one.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("typoon.api.request")

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request, call_next):
        rid = request.headers.get(_HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        ms = (time.monotonic() - t0) * 1000
        response.headers[_HEADER] = rid
        # Skip access logs for SSE — they stay open for hours.
        if not request.url.path.startswith("/api/events"):
            logger.info(
                "%s %s -> %d (%.0fms) [%s]",
                request.method, request.url.path,
                response.status_code, ms, rid,
            )
        return response
