"""Rendered page serving."""

from __future__ import annotations

import tempfile
from pathlib import Path

import bunle
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import render_key
from typoon.api.deps import get_artifact_store, get_paths, get_store, require_user
from typoon.api.routes._shared import require_project_view
from typoon.paths import Paths
from typoon.storage import Store

router = APIRouter(
    prefix="/api/projects", tags=["pages"],
    dependencies=[Depends(require_user)],
)


@router.get("/{project_id}/chapters/{chapter_id}/pages/{index}")
async def get_page(
    project_id: int,
    chapter_id: int,
    index:      int,
    user:  dict           = Depends(require_user),
    db:    Store          = Depends(get_store),
    paths: Paths          = Depends(get_paths),
    store: ArtifactStore  = Depends(get_artifact_store),
):
    """Serve a single rendered page from `render.bnl`.

    Range-streaming the archive end-to-end is an app-layer optimization;
    here we simply download the archive into a tmpfile and slice the page
    byte range out of it. For local dev with `LocalArtifactStore` this is
    a single file copy.
    """
    await require_project_view(project_id, user, db)

    state = await db.get_chapter_render_state(chapter_id)
    if state is None or not state["rendered"]:
        raise HTTPException(404, "Page not rendered yet")

    key = render_key(project_id, chapter_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "render.bnl"
        await store.get_file(key, local)
        with bunle.Reader(str(local)) as r:
            if index < 0 or index >= r.page_count:
                raise HTTPException(404, "Page index out of range")
            data = bytes(r.page(index))
            info = r.info(index)

    media_type = {"webp": "image/webp", "jpeg": "image/jpeg", "jxl": "image/jxl"}.get(
        info["format"], "application/octet-stream"
    )
    return Response(content=data, media_type=media_type)
