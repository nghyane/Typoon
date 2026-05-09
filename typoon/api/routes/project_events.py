"""Project-scoped SSE event stream.

URL: GET /api/projects/{project_id}/events?token=<jwt>

Each tab open on a project page subscribes to a Postgres LISTEN
channel scoped to that project — workers publishing per-project
events fan out only to the channel of the project they touch, so a
user viewing project 9 doesn't receive events for project 11.

Auth: EventSource can't set Authorization headers, so the JWT comes
through the `?token=` query string. We verify it on connection and
also enforce the same project-visibility rules as ordinary GET
requests via require_project_view.

Disconnect: a heartbeat every 15s lets us re-check
request.is_disconnected() and the app-level shutdown event so the
generator returns promptly when the tab closes or uvicorn starts
graceful shutdown.
"""

from __future__ import annotations

import asyncio
import json

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from typoon.adapters.channel_bus import ChannelBus, project_channel
from typoon.api.auth import verify_jwt
from typoon.api.deps import get_auth_cfg, get_bus, get_store
from typoon.api.routes._shared import require_project_view
from typoon.config import AuthConfig
from typoon.storage import Store

router = APIRouter(tags=["events"])

_HEARTBEAT_SEC = 15


@router.get("/api/projects/{project_id}/events")
async def project_event_stream(
    project_id: int,
    request: Request,
    token: str        = Query(..., description="JWT — same as Authorization header"),
    bus:   ChannelBus = Depends(get_bus),
    db:    Store      = Depends(get_store),
    cfg:   AuthConfig = Depends(get_auth_cfg),
):
    # Auth: verify token, then enforce project visibility — a token alone
    # doesn't grant access to a private project the user can't view.
    try:
        user_id, role_ids = verify_jwt(token, cfg=cfg)
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e
    user = await db.get_user(user_id)
    if user is None:
        raise HTTPException(401, "User not found")
    user["roles"] = role_ids
    await require_project_view(project_id, user, db)

    channel = project_channel(project_id)
    shutdown: asyncio.Event = request.app.state.shutdown

    async def _generate():
        async with bus.subscribe(channel) as queue:
            next_task: asyncio.Task | None = None
            shutdown_task: asyncio.Task | None = None
            try:
                while True:
                    if shutdown.is_set() or await request.is_disconnected():
                        return

                    if next_task is None:
                        next_task = asyncio.create_task(queue.get())
                    if shutdown_task is None or shutdown_task.done():
                        shutdown_task = asyncio.create_task(shutdown.wait())

                    done, _pending = await asyncio.wait(
                        {next_task, shutdown_task},
                        timeout=_HEARTBEAT_SEC,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if shutdown_task in done:
                        return

                    if next_task in done:
                        data = next_task.result()
                        next_task = None
                        yield f"data: {json.dumps(data)}\n\n"
                    else:
                        # Heartbeat — comment lines are ignored by
                        # EventSource but reset proxy idle timers and
                        # let us re-check disconnect on the next loop.
                        yield ": ping\n\n"
            finally:
                for task in (next_task, shutdown_task):
                    if task is not None and not task.done():
                        task.cancel()

    return StreamingResponse(_generate(), media_type="text/event-stream")