"""Source connector listing + URL discovery + pull.

Project creation goes through POST /api/projects (form fields). This
module owns:

  GET  /api/sources              connector catalog for the UI
  POST /api/discover             probe a URL → SourceInfo
  POST /api/projects/{id}/pull   queue chapter downloads in background

Pull is async: chapters are queued via BackgroundTasks; clients should
subscribe to /api/events to see ChapterDownloaded / ChapterFailed.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.event_bus import EventBus, EventHook
from typoon.adapters.projects import Projects
from typoon.api.deps import get_artifact_store, get_bus, get_paths, get_store
from typoon.api.models import (
    ChapterVariantOut, DiscoveredChapterOut, SourceConnectorOut, SourceInfoOut,
)
from typoon.api.routes._shared import require_project
from typoon.domain.project import DiscoveredChapter
from typoon.paths import Paths
from typoon.runs.events import Hook
from typoon.storage import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["sources"])


class DiscoverBody(BaseModel):
    url: str


class PullBody(BaseModel):
    url:      str
    from_:    float | None = None  # inclusive
    to:       float | None = None  # inclusive
    chapters: list[float] | None = None  # explicit list overrides from/to

    model_config = {"populate_by_name": True}


@router.get("/sources", response_model=list[SourceConnectorOut])
async def list_sources():
    """Connectors available for URL pulls. Used by the 'Pull from URL' UI."""
    from typoon.sources.connectors import get_connectors
    return [
        SourceConnectorOut(
            id=c.site_id,
            name=c.site_name,
            source_lang=c.source_lang,
            example_url=c.example_url,
            description=c.description,
        )
        for c in get_connectors()
    ]


@router.post("/discover", response_model=SourceInfoOut)
async def discover(body: DiscoverBody):
    from typoon.sources.connectors import get_connectors
    connector = next((c for c in get_connectors() if c.accepts(body.url)), None)
    if connector is None:
        raise HTTPException(400, f"No connector for URL: {body.url}")
    try:
        info = await connector.discover(body.url)
    except Exception as e:
        logger.exception("discover failed")
        raise HTTPException(502, f"discover failed: {e}") from e
    return SourceInfoOut(
        suggested_title=info.suggested_title,
        cover_url=info.cover_url,
        description=info.description,
        source_lang=connector.source_lang,
        chapters=[
            DiscoveredChapterOut(
                number=ch.number,
                title=ch.title,
                variants=[
                    ChapterVariantOut(id=v.id, url=v.url, group=v.group, votes=v.votes)
                    for v in ch.variants
                ],
            )
            for ch in info.chapters
        ],
    )


@router.post("/projects/{project_id}/pull", status_code=202)
async def pull(
    project_id: int,
    body:       PullBody,
    background: BackgroundTasks,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
    bus:   EventBus      = Depends(get_bus),
):
    """Pull chapters into a project (existing or freshly created)."""
    proj = await require_project(project_id, db)
    pj   = Projects(db, paths, store)
    info = await pj.discover(body.url)
    selected = _select(info.chapters, body)
    if not selected:
        raise HTTPException(400, "No chapters in the requested range")

    loop = asyncio.get_running_loop()
    hook: Hook = EventHook(bus, loop)
    background.add_task(
        _run_pull, proj["slug"], body.url, info, selected, hook,
        db, paths, store,
    )
    return {"project_id": project_id, "queued": len(selected)}


# ── Helpers ───────────────────────────────────────────────────────────


def _select(
    chapters: list[DiscoveredChapter],
    body: PullBody,
) -> list[DiscoveredChapter]:
    if body.chapters is not None:
        wanted = set(body.chapters)
        return [c for c in chapters if c.number in wanted]
    lo = body.from_ if body.from_ is not None else float("-inf")
    hi = body.to    if body.to    is not None else float("inf")
    return [c for c in chapters if lo <= c.number <= hi]


async def _run_pull(
    slug: str, url: str, info, selected, hook: Hook,
    db: Store, paths: Paths, store: ArtifactStore,
) -> None:
    try:
        await Projects(db, paths, store).pull_more(slug, url, info, selected, hook)
    except Exception:
        logger.exception("background pull failed (slug=%s)", slug)
