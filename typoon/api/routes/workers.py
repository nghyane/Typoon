"""Worker / queue introspection — Tier B dashboard endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from typoon.api.deps import get_store
from typoon.api.models import QueueStatsOut, StageStatsOut
from typoon.storage import Store

router = APIRouter(prefix="/api", tags=["workers"])


@router.get("/workers", response_model=QueueStatsOut)
async def workers(db: Store = Depends(get_store)):
    raw = await db.queue_stats()
    return QueueStatsOut(
        stages={
            stage: StageStatsOut(**counts)
            for stage, counts in raw["stages"].items()
        },
        active_workers=raw["active_workers"],
    )
