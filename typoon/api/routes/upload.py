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
import math
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from typoon.adapters.channel_bus import ChannelBus, ChannelHook
from typoon.adapters.projects import Projects
from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_storage, get_bus, get_paths, get_store, require_user
from typoon.api.models import ChapterOut
from typoon.api.routes._shared import chapter_out, require_project_owner
from typoon.paths import Paths
from typoon.runs.events import Hook
from typoon.sources.upload import (
    UnpackError, detect_kind, unpack_pdf, unpack_zip, write_image_files,
)
from typoon.storage import Store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects", tags=["upload"],
    dependencies=[Depends(require_user)],
)

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
    number:     str | None       = Form(
        None,
        description="Display chapter number (free-form: '4', '4.5', 'Extra'). "
                    "Falls back to the leading number parsed from the filename, "
                    "then to the next sequential integer in the project.",
    ),
    title:      str | None       = Form(None),
    start:      bool             = Form(
        False,
        description="When true, enqueue scan immediately so the pipeline runs "
                    "without a separate /start call. Default false: chapter "
                    "lands in `idle` and the user (or batch trigger) commits "
                    "the LLM cost explicitly. Tools that already represent a "
                    "user commitment (extension import, CLI) should pass true.",
    ),
    user:  dict            = Depends(require_user),
    db:    Store           = Depends(get_store),
    paths: Paths           = Depends(get_paths),
    stores: StorageRegistry = Depends(get_storage),
    bus:   ChannelBus       = Depends(get_bus),
):
    """Ingest one chapter from an uploaded archive, PDF, or image set.

    The display `number` is what users see. Resolution order:
      1. Form `number` (callers like the SPA dialog and the extension
         supply it explicitly).
      2. Leading numeric token in the first filename ("ch12.cbz" → "12").
      3. `max(numeric_number) + 1` over the existing chapters, or "1"
         on an empty project.

    Server assigns the internal `position` (sort key) — see
    `_resolve_chapter_position`.
    """
    await require_project_owner(project_id, user, db)
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

    # Resolve display number with a 3-step fallback so callers with no
    # filename context (e.g. extension drag-drop of unnamed pages) still
    # land on a sensible chapter without a 400. The DB column is
    # NOT NULL and the UI must always have something to print.
    raw_number = (
        (number or "").strip()
        or _number_from_filename(files[0].filename or "")
        or await _next_sequential_number(db, project_id)
    )

    loop = asyncio.get_running_loop()
    hook: Hook = ChannelHook(bus, loop)

    pj = Projects(db, paths, stores.pipeline)

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
                project_id, raw_number, pages_dir,
                title=title, hook=hook,
                # User uploaded discrete pages — never stitch them together.
                # Auto-detect would mistake a few large color images for a
                # webtoon strip and chunk it.
                strategy="one_to_one",
                start=start,
            )
        except Exception as e:
            logger.exception(
                "ingest_chapter failed (project=%s number=%s)",
                project_id, raw_number,
            )
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


_NUMBER_RE = re.compile(r"(?:^|[^\d])(\d+(?:\.\d+)?)")


def _number_from_filename(name: str) -> str:
    """Pull the first numeric token out of strings like 'ch12.cbz' → '12'.

    Returns "" when the filename has no number; the upload route then
    falls back to `_next_sequential_number`.
    """
    base = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = _NUMBER_RE.search(base)
    return m.group(1) if m else ""


async def _next_sequential_number(db: Store, project_id: int) -> str:
    """Return `floor(max numeric number) + 1`, or "1" when the project
    is empty or contains only label-only chapters (Extra/Oneshot).

    Last-resort default for uploads where neither the form `number`
    nor the filename gives us anything — better to land at "1", "2",
    "3"… than to bounce a valid pile of pages with a 400.
    """
    rows = await db.get_all_chapters(project_id)
    max_num = 0.0
    for r in rows:
        try:
            n = float(r["number"])
        except (TypeError, ValueError):
            continue
        if n > max_num:
            max_num = n
    return str(math.floor(max_num) + 1)


_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple:
    return tuple(
        int(p) if p.isdigit() else p.lower()
        for p in _NATURAL_SPLIT.split(name)
    )
