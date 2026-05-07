"""SSE event stream — supports reconnect via Last-Event-ID."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from typoon.adapters.event_bus import EventBus
from typoon.api.deps import get_bus

router = APIRouter(tags=["events"])


@router.get("/api/events")
async def event_stream(request: Request, bus: EventBus = Depends(get_bus)):
    last_id = request.headers.get("last-event-id", "0-0")

    async def _generate():
        async for msg_id, data in bus.subscribe(last_id):
            if await request.is_disconnected():
                break
            yield f"id: {msg_id}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
