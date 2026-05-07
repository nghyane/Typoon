"""SSE event stream — supports reconnect via Last-Event-ID.

Auth note. EventSource doesn't let the page set Authorization headers,
so the token comes in via the `?token=` query param instead. The same
JWT works — we verify it once at connection time and attach the user
to the stream.

Disconnect handling. EventBus.subscribe() blocks at queue.get() until
the next event arrives. If the client disconnects mid-wait we must drop
the subscriber promptly, otherwise dev hot-reloads accumulate orphaned
bus subscribers and the API process slows to a crawl.

Strategy: race anext(events) against a periodic disconnect probe. The
heartbeat doubles as a keep-alive so proxies don't idle-close the
connection.
"""

from __future__ import annotations

import asyncio
import json

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from typoon.adapters.event_bus import EventBus
from typoon.api.auth import verify_jwt
from typoon.api.deps import get_auth_cfg, get_bus, get_store
from typoon.config import AuthConfig
from typoon.storage import Store

router = APIRouter(tags=["events"])

_HEARTBEAT_SEC = 15


@router.get("/api/events")
async def event_stream(
    request: Request,
    token:   str        = Query(..., description="JWT — same as Authorization header"),
    bus:     EventBus   = Depends(get_bus),
    db:      Store      = Depends(get_store),
    cfg:     AuthConfig = Depends(get_auth_cfg),
):
    # Verify token before opening the stream. Failures must be 401, not
    # mid-stream errors.
    try:
        user_id = verify_jwt(token, cfg=cfg)
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e
    if await db.get_user(user_id) is None:
        raise HTTPException(401, "User not found")

    last_id = request.headers.get("last-event-id", "0")

    async def _generate():
        events = bus.subscribe(last_id).__aiter__()
        next_task: asyncio.Task | None = None
        try:
            while True:
                if await request.is_disconnected():
                    return

                if next_task is None:
                    next_task = asyncio.create_task(events.__anext__())

                done, _pending = await asyncio.wait(
                    {next_task}, timeout=_HEARTBEAT_SEC,
                )

                if next_task in done:
                    try:
                        msg_id, data = next_task.result()
                    except StopAsyncIteration:
                        return
                    next_task = None
                    yield f"id: {msg_id}\ndata: {json.dumps(data)}\n\n"
                else:
                    # Heartbeat — comment lines are ignored by EventSource
                    # but reset proxy idle timers and let us re-check
                    # disconnect on the next loop.
                    yield ": ping\n\n"
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()
                try:
                    await next_task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    pass
            await events.aclose()

    return StreamingResponse(_generate(), media_type="text/event-stream")
