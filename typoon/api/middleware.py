"""Request-ID middleware (pure ASGI).

Stamps every incoming request with a short id, exposes it via
`request.state.request_id`, and echoes it back in the `X-Request-ID`
response header. Logs surface the same id so a client-side error
report can be matched to server logs without greping by timestamp.

If the client supplied an inbound `X-Request-ID` we trust it (closed
community deploy — no spoof concern); otherwise we mint one.

Implementation note: Starlette's `BaseHTTPMiddleware` buffers the
response body through an `anyio` memory stream, which breaks
streaming responses (SSE) on graceful shutdown — uvicorn cancels the
underlying task and the buffered receive raises `WouldBlock` /
`CancelledError`. We use a pure-ASGI middleware so the response is
forwarded as-is, send-by-send, no extra task group.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("typoon.api.request")

_HEADER = "X-Request-ID"
_HEADER_BYTES = _HEADER.lower().encode("latin-1")


class RequestIDMiddleware:
    """Pure-ASGI middleware. Safe for SSE & streaming bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = _extract_request_id(scope) or uuid.uuid4().hex[:12]
        # Mirror BaseHTTPMiddleware behaviour: expose via request.state.
        scope.setdefault("state", {})["request_id"] = rid

        t0 = time.monotonic()
        status_code = 500  # overwritten by response.start
        rid_bytes = rid.encode("latin-1")

        async def send_with_rid(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Append (don't dedupe) — handlers shouldn't be
                # setting this themselves, but if they do, the
                # last value the client sees is ours.
                headers = list(message.get("headers", []))
                headers.append((_HEADER_BYTES, rid_bytes))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_rid)
        except Exception:
            logger.exception(
                "request failed",
                extra={
                    "request_id": rid,
                    "method": scope.get("method"),
                    "path":   scope.get("path"),
                },
            )
            raise
        finally:
            path = scope.get("path", "")
            # SSE streams stay open for hours — one access log per
            # connect would dwarf real traffic. Skip them.
            if not path.startswith("/api/events"):
                ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "%s %s -> %d (%.0fms) [%s]",
                    scope.get("method"), path, status_code, ms, rid,
                )


def _extract_request_id(scope: Scope) -> str | None:
    for k, v in scope.get("headers", ()):
        if k == _HEADER_BYTES:
            try:
                return v.decode("latin-1")
            except UnicodeDecodeError:
                return None
    return None
