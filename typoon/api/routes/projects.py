"""Project + chapter CRUD routes."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.adapters.projects import Projects
from typoon.api.deps import (
    get_storage, get_paths, get_store, require_user,
)
from typoon.api.models import ChapterOut, ProjectOut
from typoon.api.routes._shared import (
    chapter_out, require_chapter,
    require_project_owner, require_project_view,
)
from typoon.paths import Paths, ProjectPaths, slugify
from typoon.storage import Store

router = APIRouter(
    prefix="/api/projects", tags=["projects"],
    dependencies=[Depends(require_user)],
)


# ── Request bodies ────────────────────────────────────────────────────


class CreateProjectBody(BaseModel):
    title:       str
    description: str | None = None
    source_lang: str        = "en"
    target_lang: str        = "vi"


class SettingsBody(BaseModel):
    target_lang: str | None  = None
    settings:    dict | None = None  # arbitrary JSON overrides
    title:       str | None  = None
    description: str | None  = None
    shared:      bool | None = None


# ── Projects ──────────────────────────────────────────────────────────


_VALID_FILTERS = {"all", "mine", "pinned", "community"}


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    filter: str = Query("all"),
    user:   dict = Depends(require_user),
    db:     Store = Depends(get_store),
):
    """List projects visible to the current user.

    filter:
      mine       owned by me
      pinned     bookmarked by me
      community  shared by others
      all        owned + shared (default sidebar landing)
    """
    if filter not in _VALID_FILTERS:
        raise HTTPException(400, f"filter must be one of {sorted(_VALID_FILTERS)}")
    rows = await db.list_projects(viewer_id=user["id"], filter=filter)
    return [ProjectOut.from_row(p, viewer_id=user["id"]) for p in rows]


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body:  CreateProjectBody,
    user:  dict = Depends(require_user),
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    """Create an empty project owned by the current user."""
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "title required")

    slug = slugify(title)
    pid  = await db.get_or_create_project(
        slug=slug, title=title,
        source_lang=body.source_lang, target_lang=body.target_lang,
        owner_id=user["id"],
    )
    if body.description:
        await db.update_project_metadata(pid, description=body.description)
    ProjectPaths(paths.projects, slug).ensure()

    proj = await db.get_project(pid)
    return ProjectOut.from_row(proj, viewer_id=user["id"])


@router.post("/{project_id}/cover", response_model=ProjectOut)
async def upload_cover(
    project_id: int,
    file:       UploadFile = File(...),
    user:  dict  = Depends(require_user),
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    """Upload a cover image. Owner only."""
    proj = await require_project_owner(project_id, user, db)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "cover must be an image")

    raw = await file.read()
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e

    proj_paths = ProjectPaths(paths.projects, proj["slug"])
    proj_paths.ensure()
    img.save(proj_paths.cover, format="JPEG", quality=88)
    await db.update_project_metadata(project_id, cover_path=str(proj_paths.cover))

    proj = await db.get_project(project_id)
    return ProjectOut.from_row(proj, viewer_id=user["id"])


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    proj = await require_project_view(project_id, user, db)
    out  = ProjectOut.from_row(proj, viewer_id=user["id"])
    out.is_pinned = await db.is_pinned(user["id"], project_id)
    return out


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
):
    proj = await require_project_owner(project_id, user, db)
    await db.delete_project(project_id)
    shutil.rmtree(ProjectPaths(paths.projects, proj["slug"]).root, ignore_errors=True)
    shutil.rmtree(paths.artifacts / "p" / str(project_id), ignore_errors=True)


# ── Pin (bookmark) ────────────────────────────────────────────────────


@router.post("/{project_id}/pin", status_code=204)
async def pin_project(
    project_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Bookmark a project. User must be able to view it
    (owner or project shared)."""
    await require_project_view(project_id, user, db)
    await db.pin_project(user["id"], project_id)


@router.delete("/{project_id}/pin", status_code=204)
async def unpin_project(
    project_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    await db.unpin_project(user["id"], project_id)


# ── Settings ──────────────────────────────────────────────────────────


@router.get("/{project_id}/settings")
async def get_settings(
    project_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    proj = await require_project_view(project_id, user, db)
    return {
        "project_id":  project_id,
        "target_lang": proj["target_lang"],
        "title":       proj["title"],
        "description": proj.get("description"),
        "shared":      bool(proj.get("shared")),
        "is_owner":    proj.get("owner_id") == user["id"],
        "settings":    await db.get_project_settings(project_id),
    }


@router.patch("/{project_id}/settings")
async def patch_settings(
    project_id: int,
    body:       SettingsBody,
    user:       dict  = Depends(require_user),
    db:         Store = Depends(get_store),
):
    """Update project settings. Owner only."""
    await require_project_owner(project_id, user, db)
    await db.update_project_metadata(
        project_id,
        title=body.title,
        description=body.description,
        target_lang=body.target_lang,
        settings=body.settings,
        shared=body.shared,
    )
    proj = await db.get_project(project_id)
    return {
        "project_id":  project_id,
        "target_lang": proj["target_lang"],
        "title":       proj["title"],
        "description": proj.get("description"),
        "shared":      bool(proj.get("shared")),
        "is_owner":    True,
        "settings":    await db.get_project_settings(project_id),
    }


# ── Chapters ──────────────────────────────────────────────────────────


def _archive_url(
    stores: StorageRegistry, row: dict,
) -> str | None:
    """Build the public archive URL for a chapter row.

    Returns None until the chapter has both `rendered=true` and an
    archive locator persisted. URL build dispatches by the row's
    `archive_backend`, so chapters rendered against an old primary
    keep working after the operator switches the writer.

    The version query string is the row's `updated_at` so a re-render
    busts the CDN cache. The path is stable for cache-key purposes —
    only the query changes.
    """
    if row.get("state") != "done":
        return None
    backend = row.get("archive_backend")
    locator = row.get("archive_locator")
    if not backend or not locator:
        return None
    return stores.reader(backend).url(
        locator, version=str(row.get("updated_at") or ""),
    )


@router.get("/{project_id}/chapters", response_model=list[ChapterOut])
async def list_chapters(
    project_id: int,
    user:   dict                  = Depends(require_user),
    db:     Store                 = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    await require_project_view(project_id, user, db)
    rows = await db.get_chapters_with_status(project_id)
    return [chapter_out(r, archive_url=_archive_url(stores, r)) for r in rows]


@router.get("/{project_id}/chapters/{chapter_id}", response_model=ChapterOut)
async def get_chapter(
    project_id: int,
    chapter_id: int,
    user:   dict                  = Depends(require_user),
    db:     Store                 = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    await require_project_view(project_id, user, db)
    data = await db.get_chapter_with_status(chapter_id, project_id)
    if data is None:
        raise HTTPException(404, "Chapter not found")
    return chapter_out(data, archive_url=_archive_url(stores, data))


@router.delete("/{project_id}/chapters/{chapter_id}", status_code=204)
async def delete_chapter(
    project_id: int,
    chapter_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    paths:  Paths           = Depends(get_paths),
    stores: StorageRegistry = Depends(get_storage),
):
    await require_project_owner(project_id, user, db)
    ch = await require_chapter(project_id, chapter_id, db)
    archive_backend = ch.get("archive_backend")
    archive_locator = ch.get("archive_locator")

    deleted = await db.delete_chapter(chapter_id)
    if not deleted:
        raise HTTPException(404, "Chapter not found")

    from typoon.adapters.chapter_archive import prepared_key, masks_key
    # Pipeline blobs (prepared, masks) live on the worker pipeline store.
    await stores.pipeline.delete(prepared_key(project_id, chapter_id))
    await stores.pipeline.delete(masks_key(project_id, chapter_id))
    # Public render archive: dispatch to whichever backend wrote it.
    if archive_backend and archive_locator:
        try:
            await stores.reader(archive_backend).delete(archive_locator)
        except RuntimeError:
            pass  # backend no longer configured — orphan, nothing to do


@router.post("/{project_id}/chapters/{chapter_id}/redo", response_model=ChapterOut)
async def redo_chapter(
    project_id: int,
    chapter_id: int,
    user:   dict                  = Depends(require_user),
    db:     Store                 = Depends(get_store),
    paths:  Paths                 = Depends(get_paths),
    stores: StorageRegistry = Depends(get_storage),
):
    proj = await require_project_owner(project_id, user, db)
    ch   = await require_chapter(project_id, chapter_id, db)
    await Projects(db, paths, stores.pipeline).redo(proj["slug"], [ch["id"]])
    data = await db.get_chapter_with_status(chapter_id, project_id)
    return chapter_out(data, archive_url=_archive_url(stores, data))


# ── Manual pipeline trigger ───────────────────────────────────────────
#
# Upload defaults to `start=false` so users can review chapters before
# committing LLM cost. These endpoints kick the scan stage on idle
# chapters; they're the web UI counterpart to `redo` (which is for
# rerunning chapters that already finished or errored out).


class StartChaptersBody(BaseModel):
    chapter_ids: list[int]


@router.post("/{project_id}/chapters/{chapter_id}/start", response_model=ChapterOut)
async def start_chapter(
    project_id: int,
    chapter_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    paths:  Paths           = Depends(get_paths),
    stores: StorageRegistry = Depends(get_storage),
):
    """Enqueue scan for a single idle chapter. No-op if it's already
    running, queued, or finished — caller should use `/redo` to rerun
    a non-idle chapter."""
    await require_project_owner(project_id, user, db)
    await require_chapter(project_id, chapter_id, db)
    await Projects(db, paths, stores.pipeline).start_chapters(
        project_id, [chapter_id],
    )
    data = await db.get_chapter_with_status(chapter_id, project_id)
    return chapter_out(data, archive_url=_archive_url(stores, data))


@router.post("/{project_id}/chapters/start")
async def start_chapters(
    project_id: int,
    body:   StartChaptersBody,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    paths:  Paths           = Depends(get_paths),
    stores: StorageRegistry = Depends(get_storage),
):
    """Batch trigger for the selection-bar action. Returns the count
    actually started (idle chapters only); ids that were already in
    flight or done are silently skipped so a partial selection still
    succeeds."""
    await require_project_owner(project_id, user, db)
    started = await Projects(db, paths, stores.pipeline).start_chapters(
        project_id, body.chapter_ids,
    )
    return {"started": started, "total": len(body.chapter_ids)}


# ── Chapter brief ─────────────────────────────────────────────────────


@router.get("/{project_id}/chapters/{chapter_id}/brief")
async def get_brief(
    project_id: int,
    chapter_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    await require_project_view(project_id, user, db)
    await require_chapter(project_id, chapter_id, db)
    brief = await db.get_chapter_brief(chapter_id)
    if brief is None:
        raise HTTPException(404, "Brief not generated yet")
    return brief
