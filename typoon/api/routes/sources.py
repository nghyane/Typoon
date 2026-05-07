"""Source discovery + pull routes.

Discovery is a foreground call (returns SourceInfo). Pull is async-ish:
it kicks off prepare_chapter_to_archive in the background and returns
immediately with a pulled-count summary. Workers pick up the chapter
once prepare is done.
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
    ChapterVariantOut, DiscoveredChapterOut, ProjectOut, SourceInfoOut,
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


class PullNewBody(BaseModel):
    url:         str
    target_lang: str = "vi"
    from_:       float | None = None  # inclusive
    to:          float | None = None  # inclusive
    chapters:    list[float] | None = None  # explicit list overrides from/to

    model_config = {"populate_by_name": True}


class PullMoreBody(BaseModel):
    url:      str
    from_:    float | None = None
    to:       float | None = None
    chapters: list[float] | None = None


@router.post("/discover", response_model=SourceInfoOut)
async def discover(body: DiscoverBody, db: Store = Depends(get_store)):
    """Probe a manga source URL. Returns title, cover, synopsis, chapter list.

    Project creation is a separate step (POST /api/projects/pull).
    """
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


@router.post("/projects/pull", response_model=ProjectOut, status_code=202)
async def pull_new_project(
    body:       PullNewBody,
    background: BackgroundTasks,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
    bus:   EventBus      = Depends(get_bus),
):
    """Discover + create project + start downloading the selected range.

    Returns 202 with the freshly-created ProjectOut. Downloads happen in
    a background task; clients should subscribe to /api/events to see
    ChapterDownloaded / ChapterFailed.
    """
    pj = Projects(db, paths, store)
    info = await pj.discover(body.url)
    selected = _select(info.chapters, body)
    if not selected:
        raise HTTPException(400, "No chapters in the requested range")

    loop = asyncio.get_running_loop()
    hook: Hook = EventHook(bus, loop)

    slug = await pj.create_project(info, body.url, body.target_lang)
    proj = await db.get_project_by_slug(slug)
    background.add_task(_run_pull_more, slug, body.url, info, selected, hook,
                        db, paths, store)
    return ProjectOut.from_row(proj)


@router.post("/projects/{project_id}/pull", status_code=202)
async def pull_more(
    project_id: int,
    body:       PullMoreBody,
    background: BackgroundTasks,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
    bus:   EventBus      = Depends(get_bus),
):
    """Pull more chapters into an existing project."""
    proj = await require_project(project_id, db)
    pj   = Projects(db, paths, store)
    info = await pj.discover(body.url)
    selected = _select(info.chapters, body)
    if not selected:
        raise HTTPException(400, "No chapters in the requested range")

    loop = asyncio.get_running_loop()
    hook: Hook = EventHook(bus, loop)
    background.add_task(_run_pull_more, proj["slug"], body.url, info, selected, hook,
                        db, paths, store)
    return {"project_id": project_id, "queued": len(selected)}


# ── Helpers ───────────────────────────────────────────────────────────


def _select(
    chapters: list[DiscoveredChapter],
    body: PullNewBody | PullMoreBody,
) -> list[DiscoveredChapter]:
    if body.chapters is not None:
        wanted = set(body.chapters)
        return [c for c in chapters if c.number in wanted]
    lo = body.from_ if body.from_ is not None else float("-inf")
    hi = body.to    if body.to    is not None else float("inf")
    return [c for c in chapters if lo <= c.number <= hi]


async def _run_pull_more(
    slug: str, url: str, info, selected, hook: Hook,
    db: Store, paths: Paths, store: ArtifactStore,
) -> None:
    try:
        await Projects(db, paths, store).pull_more(slug, url, info, selected, hook)
    except Exception:
        logger.exception("background pull_more failed (slug=%s)", slug)
