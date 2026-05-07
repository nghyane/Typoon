"""Chapter ingestion via file upload.

Accepts PDF, ZIP/CBZ, or a multi-image set. The bytes are unpacked into
a temp folder and handed to `Projects.ingest_chapter` which packs the
chapter into prepared.bnl and enqueues `scan`.

This is the only ingestion path on the server — no scrapers, no
cookies, no CF dance. Discord bot / browser extension scrape on the
user's side and POST here.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.event_bus import EventBus, EventHook
from typoon.adapters.projects import Projects
from typoon.api.deps import get_artifact_store, get_bus, get_paths, get_store
from typoon.api.models import ChapterOut
from typoon.api.routes._shared import chapter_out, require_project
from typoon.paths import Paths
from typoon.runs.events import Hook
from typoon.sources.upload import (
    UnpackError, detect_kind, unpack_pdf, unpack_zip, write_image_files,
)
from typoon.storage import Store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["upload"])

# Hard cap to keep a runaway upload from exhausting tmp. 1.5 GiB easily
# fits a 200-page Kindle PDF (~700KB/page * 200 = ~140MB) plus headroom.
_MAX_UPLOAD_BYTES = 1500 * 1024 * 1024


@router.post(
    "/{project_id}/chapters/upload",
    response_model=ChapterOut,
    status_code=202,
)
async def upload_chapter(
    project_id: int,
    files:      list[UploadFile] = File(..., description="ZIP/CBZ/PDF or multiple images"),
    idx:        float | None     = Form(None, description="Chapter number; auto from filename if omitted"),
    title:      str | None       = Form(None),
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
    bus:   EventBus      = Depends(get_bus),
):
    """Ingest one chapter from an uploaded archive, PDF, or image set.

    Single archive (ZIP/CBZ/PDF) → idx defaults to a number parsed from
    the filename; multi-image upload → idx must be supplied or we
    fall back to filename of the first image.
    """
    proj = await require_project(project_id, db)
    files = [f for f in files if f.filename]
    if not files:
        raise HTTPException(400, "No files in upload")

    # Decide a single ingestion shape.
    if len(files) == 1:
        first = files[0]
        try:
            kind = detect_kind(first.filename or "", first.content_type)
        except UnpackError as e:
            raise HTTPException(400, str(e)) from e
    else:
        kind = "image"  # multi-file upload is always treated as an image set

    raw_idx = idx if idx is not None else _idx_from_filename(files[0].filename or "")
    if raw_idx is None:
        raise HTTPException(
            400, "Could not infer chapter number — supply idx in form data",
        )

    loop = asyncio.get_running_loop()
    hook: Hook = EventHook(bus, loop)

    pj = Projects(db, paths, store)

    # Unpack to temp, ingest, return chapter status.
    with tempfile.TemporaryDirectory(prefix="typoon-upload-") as tmp:
        pages_dir = Path(tmp) / "pages"

        try:
            n = await asyncio.to_thread(_unpack, files, kind, pages_dir)
        except UnpackError as e:
            raise HTTPException(400, str(e)) from e

        if n == 0:
            raise HTTPException(400, "Upload contained no pages")

        try:
            chapter_id = await pj.ingest_chapter(
                project_id, raw_idx, pages_dir,
                title=title, hook=hook,
                # User uploaded discrete pages — never stitch them together.
                # Auto-detect would mistake a few large color images for a
                # webtoon strip and chunk it.
                strategy="one_to_one",
            )
        except Exception as e:
            logger.exception("ingest_chapter failed (project=%s idx=%s)", project_id, raw_idx)
            raise HTTPException(500, f"ingest failed: {e}") from e

    data = await db.get_chapter_with_status(chapter_id, project_id)
    if data is None:
        raise HTTPException(500, "Chapter created but lookup failed")
    return chapter_out(data)


# ── Helpers ───────────────────────────────────────────────────────────


def _unpack(
    files: list[UploadFile], kind: str, dest: Path,
) -> int:
    """Synchronous unpack helper; runs in a worker thread."""
    if kind == "pdf":
        data = _read_capped(files[0])
        return unpack_pdf(data, dest)
    if kind == "zip":
        data = _read_capped(files[0])
        return unpack_zip(data, dest)
    # image set — natural-sort by original filename
    pairs = [(f.filename or f"page-{i}", _read_capped(f))
             for i, f in enumerate(files)]
    pairs.sort(key=lambda p: _natural_key(p[0]))
    return write_image_files(pairs, dest)


def _read_capped(f: UploadFile) -> bytes:
    """Read entire UploadFile body, refuse anything over the cap."""
    f.file.seek(0)
    data = f.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {_MAX_UPLOAD_BYTES // (1024*1024)} MiB)")
    return data


_IDX_RE = re.compile(r"(?:^|[^\d])(\d+(?:\.\d+)?)")


def _idx_from_filename(name: str) -> float | None:
    """Pull a chapter number out of strings like 'ch12.cbz', 'chapter-7.pdf'."""
    base = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = _IDX_RE.search(base)
    return float(m.group(1)) if m else None


_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple:
    return tuple(
        int(p) if p.isdigit() else p.lower()
        for p in _NATURAL_SPLIT.split(name)
    )
