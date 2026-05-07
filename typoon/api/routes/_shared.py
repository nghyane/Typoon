"""Shared route helpers."""

from __future__ import annotations

from fastapi import HTTPException

from typoon.api.models import ChapterOut, Progress
from typoon.storage import Store


async def require_project(project_id: int, db: Store) -> dict:
    """Resolve project by id or 404. Permission-agnostic — caller must
    pair with `_require_view`/`_require_owner` for access control."""
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    return proj


async def require_project_view(project_id: int, user: dict, db: Store) -> dict:
    """Project visible to user (owner OR shared). 404 hides existence
    of private projects from non-owners."""
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    if proj.get("owner_id") != user["id"] and not proj.get("shared"):
        raise HTTPException(404, "Project not found")
    return proj


async def require_project_owner(project_id: int, user: dict, db: Store) -> dict:
    """Project must be owned by user. Returns project row.

    NULL owner_id (legacy projects pre-RFC-006) treated as no-owner —
    nobody can mutate them via the API. Admin can backfill owner_id via
    direct SQL or future admin endpoint.
    """
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    if proj.get("owner_id") != user["id"]:
        raise HTTPException(403, "Only the owner can do this")
    return proj


async def require_chapter(project_id: int, chapter_id: int, db: Store) -> dict:
    ch = await db.get_chapter(chapter_id)
    if ch is None or ch["project_id"] != project_id:
        raise HTTPException(404, "Chapter not found")
    return ch


def chapter_out(data: dict) -> ChapterOut:
    page_count = int(data.get("page_count") or 0)
    progress_data = data.get("progress")
    progress = (
        Progress(
            stage=progress_data.get("stage") or data.get("stage") or "",
            page_index=int(progress_data.get("page_index") or 0),
            page_total=int(progress_data.get("page_total") or page_count),
        )
        if progress_data and data["state"] == "running"
        else None
    )
    return ChapterOut(
        chapter_id=data["chapter_id"],
        project_id=data["project_id"],
        idx=data["idx"],
        title=data.get("title"),
        state=data["state"],
        stage=data.get("stage") or "",
        page_count=page_count,
        error=data.get("error") or "",
        updated_at=data.get("updated_at"),
        progress=progress,
    )
