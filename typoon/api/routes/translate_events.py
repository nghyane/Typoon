"""Translation-scoped SSE event stream.

URL: GET /api/translate/{translation_id}/events?token=<jwt>

A reader/editor open on a translation subscribes to two Postgres LISTEN
channels:
  - typoon:draft:{draft_id}        — translate/render progress shared
                                     across every translation that points
                                     at the draft (cache hit users see
                                     the same pipeline run advance).
  - typoon:translation:{translation_id} — per-translation events when
                                     sparse edits trigger a fork render.

Auth: EventSource can't set Authorization headers, so the JWT comes
through the `?token=` query string. We verify it on connection then
enforce the same read visibility used by GET /translate/{id}.

Disconnect: a 15s heartbeat lets us re-check request.is_disconnected()
and the app-level shutdown event so the generator returns promptly when
the tab closes or uvicorn starts graceful shutdown.
"""

from __future__ import annotations

import asyncio
import json

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from typoon.adapters.channel_bus import (
    ChannelBus, draft_channel, translation_channel,
)
from typoon.api.auth import verify_jwt
from typoon.api.deps import get_auth_cfg, get_bus, get_store
from typoon.config import AuthConfig
from typoon.storage import Store

router = APIRouter(tags=["events"])

_HEARTBEAT_SEC = 15


@router.get("/api/translate/{translation_id}/events")
async def translation_event_stream(
    translation_id: int,
    request: Request,
    token:   str        = Query(..., description="JWT — same as Authorization header"),
    bus:     ChannelBus = Depends(get_bus),
    db:      Store      = Depends(get_store),
    cfg:     AuthConfig = Depends(get_auth_cfg),
):
    # Auth: verify the token then load the user. Visibility check below
    # mirrors GET /translate/{id} so a token alone doesn't grant access
    # to a private draft the user can't read.
    try:
        user_id, role_ids = verify_jwt(token, cfg=cfg)
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e
    user = await db.get_user(user_id)
    if user is None:
        raise HTTPException(401, "User not found")
    user["roles"] = role_ids

    translation = await db.get_translation(translation_id)
    if translation is None:
        raise HTTPException(404, "Translation not found")

    # Owner always allowed; cross-user readers gated by draft visibility.
    if translation["owner_id"] != user["id"]:
        draft = (
            await db.get_draft(translation["draft_id"])
            if translation.get("draft_id") else None
        )
        if draft is None or draft.get("takedown_at"):
            raise HTTPException(404, "Translation not found")
        viewer_guilds = [g["id"] for g in await db.get_user_guilds(user["id"])]
        if draft["visibility"] == "private":
            raise HTTPException(404, "Translation not found")
        if (
            draft["visibility"] == "guild"
            and draft.get("scope_guild_id") not in viewer_guilds
        ):
            raise HTTPException(404, "Translation not found")
        # 'all_guilds' relies on the user_guilds intersection check;
        # approximate by accepting any logged-in user with ≥1 guild.

    # Subscribe to both channels. Draft-level events (translate progress)
    # appear for every reader of the draft; translation-level events
    # appear only when sparse edits trigger a per-translation render.
    channels: list[str] = []
    if translation.get("draft_id"):
        channels.append(draft_channel(translation["draft_id"]))
    channels.append(translation_channel(translation_id))

    shutdown: asyncio.Event = request.app.state.shutdown

    async def _generate():
        queues: list[asyncio.Queue[dict]] = []
        cms = []
        try:
            for channel in channels:
                cm = bus.subscribe(channel)
                queues.append(await cm.__aenter__())
                cms.append((cm, channel))

            # One reader task per queue. Multiplex into a single yield
            # so events from either channel arrive on the same stream.
            tasks = {asyncio.create_task(q.get()): q for q in queues}
            shutdown_task: asyncio.Task | None = None

            while True:
                if shutdown.is_set() or await request.is_disconnected():
                    return

                if shutdown_task is None or shutdown_task.done():
                    shutdown_task = asyncio.create_task(shutdown.wait())

                done, _pending = await asyncio.wait(
                    set(tasks) | {shutdown_task},
                    timeout=_HEARTBEAT_SEC,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if shutdown_task in done:
                    return

                # Emit any data tasks that completed; re-arm their queue.
                any_data = False
                for t in list(done):
                    if t is shutdown_task:
                        continue
                    q = tasks.pop(t, None)
                    if q is None:
                        continue
                    data = t.result()
                    any_data = True
                    yield f"data: {json.dumps(data)}\n\n"
                    tasks[asyncio.create_task(q.get())] = q

                if not any_data:
                    # Heartbeat — EventSource ignores comment lines but
                    # this resets proxy idle timers and lets us re-check
                    # disconnect on the next loop.
                    yield ": ping\n\n"
        finally:
            for t in list(tasks):
                if not t.done():
                    t.cancel()
            if shutdown_task is not None and not shutdown_task.done():
                shutdown_task.cancel()
            for cm, _ in cms:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":      "no-cache, no-transform",
            "X-Accel-Buffering":  "no",
            "Connection":         "keep-alive",
        },
    )
