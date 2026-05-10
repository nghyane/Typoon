"""Chapter ingestion via browser-direct multipart upload.

Flow:

  1. client → POST upload-init { byte_size }
     server → quota check
            → inbox.create_multipart(tmp_id, part_count, part_size)
            → returns { tmp_id, upload_id, parts: [{number, url}], part_size }

  2. client → PUT each part to its presigned URL (parallel, e.g. 4
              concurrent), reads `ETag` from each PUT response

  3. client → POST upload-finalize {
                tmp_id, upload_id, parts: [{number, etag}],
                number?, title?,
              }
     server → quota consume
            → Projects.queue_chapter   (insert chapter + persist
                                        inbox handle + enqueue prepare)
            → returns 202 ChapterOut (state="pending"; prepare worker
              picks up immediately)

  X. client may POST upload-abort to free a half-uploaded prefix when
     the user cancels. The bucket lifecycle rule is the safety net for
     anything that slips through.

The actual zip download / unpack / prepare / bunle.pack happens in the
prepare worker (`workers.loop._run_prepare`), not in this request
handler. That keeps the API hot path under ~100ms regardless of
chapter size.

LocalInbox dev backend simulates multipart by routing PUTs back at a
sibling endpoint (`PUT /api/_inbox/{tmp_id}/{upload_id}/{n}`) so the
client SDK has one code path in dev and prod.
"""

from __future__ import annotations

import logging
import math
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from typoon.adapters.inbox import (
    DEFAULT_PART_SIZE, ChapterInbox, CompletedPart, InboxHandle, LocalInbox,
)
from typoon.adapters.projects import Projects
from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import (
    get_auth_cfg, get_config, get_inbox, get_paths, get_storage,
    get_store, require_user,
)
from typoon.api.models import ChapterOut
from typoon.api.quota import enforce_chapter_quota, record_chapter_consume
from typoon.api.routes._shared import chapter_out, require_project_owner
from typoon.config import AuthConfig, Config
from typoon.paths import Paths
from typoon.storage import Store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects", tags=["upload"],
    dependencies=[Depends(require_user)],
)

# Hard cap on declared upload size — refuse upload-init outright when a
# client claims a chapter zip larger than this. 1.5 GiB easily fits a
# 200-page lossy-WebP webtoon plus headroom.
_MAX_UPLOAD_BYTES = 1500 * 1024 * 1024

# Part-count ceiling. R2 / S3 allow up to 10000; we cap lower so a
# misbehaving client can't tie up presigning CPU. With 8 MiB parts
# this still permits 8 GiB total — well above _MAX_UPLOAD_BYTES.
_MAX_PARTS = 1024


# ── Schemas ───────────────────────────────────────────────────────


class UploadInitBody(BaseModel):
    byte_size: int = Field(ge=1, le=_MAX_UPLOAD_BYTES)


class InitPart(BaseModel):
    number: int
    url:    str


class UploadInitOut(BaseModel):
    tmp_id:     str
    upload_id:  str
    parts:      list[InitPart]
    part_size:  int
    expires_in: int


class FinalizePart(BaseModel):
    number: int = Field(ge=1, le=_MAX_PARTS)
    etag:   str = Field(min_length=1, max_length=128)


class UploadFinalizeBody(BaseModel):
    tmp_id:    str = Field(min_length=8, max_length=64)
    upload_id: str = Field(min_length=1, max_length=512)
    parts:     list[FinalizePart] = Field(min_length=1, max_length=_MAX_PARTS)
    number:    str | None = None
    title:     str | None = None


class UploadAbortBody(BaseModel):
    tmp_id:    str = Field(min_length=8, max_length=64)
    upload_id: str = Field(min_length=1, max_length=512)


# ── Routes ────────────────────────────────────────────────────────


@router.post(
    "/{project_id}/chapters/upload-init",
    response_model=UploadInitOut,
)
async def upload_init(
    project_id: int,
    body:    UploadInitBody,
    user:    dict          = Depends(require_user),
    db:      Store         = Depends(get_store),
    cfg:     Config        = Depends(get_config),
    auth:    AuthConfig    = Depends(get_auth_cfg),
    inbox:   ChapterInbox  = Depends(get_inbox),
):
    """Begin a multipart upload for one chapter zip.

    Quota is enforced **before** any presigned URL leaves the server,
    so a maxed-out user gets 429 instantly instead of after PUTting
    100 MB of data. Race window between init and finalize is closed
    by re-checking + consuming at finalize time.
    """
    await require_project_owner(project_id, user, db)
    await enforce_chapter_quota(user, db, cfg.rate_limit, auth, count=1)

    tmp_id = secrets.token_urlsafe(16)

    # Round up to part count — last part may be smaller.
    part_count = max(1, math.ceil(body.byte_size / DEFAULT_PART_SIZE))
    if part_count > _MAX_PARTS:
        raise HTTPException(
            413,
            f"Upload too large ({body.byte_size} bytes; max "
            f"{_MAX_PARTS * DEFAULT_PART_SIZE} bytes).",
        )

    upload_id, urls = await inbox.create_multipart(
        tmp_id=tmp_id, part_count=part_count, part_size=DEFAULT_PART_SIZE,
    )

    return UploadInitOut(
        tmp_id=tmp_id,
        upload_id=upload_id,
        parts=[InitPart(number=p.number, url=p.url) for p in urls],
        part_size=DEFAULT_PART_SIZE,
        expires_in=3600,  # PRESIGN_TTL_SECONDS — keep clients in sync
    )


@router.post(
    "/{project_id}/chapters/upload-finalize",
    response_model=ChapterOut,
    status_code=202,
)
async def upload_finalize(
    project_id: int,
    body:    UploadFinalizeBody,
    user:    dict             = Depends(require_user),
    db:      Store            = Depends(get_store),
    paths:   Paths            = Depends(get_paths),
    stores:  StorageRegistry  = Depends(get_storage),
    cfg:     Config           = Depends(get_config),
    auth:    AuthConfig       = Depends(get_auth_cfg),
    inbox:   ChapterInbox     = Depends(get_inbox),
):
    """Persist the multipart handle, enqueue prepare, return 202.

    The actual heavy lifting (complete_multipart on the inbox, fetch
    the zip, unpack, prepare, pack prepared.bnl) is deferred to the
    prepare worker. This keeps the request short (~50ms) so the
    Cloudflare tunnel and the browser don't have to hold a connection
    open for the full ingest cycle.
    """
    await require_project_owner(project_id, user, db)

    # Quota commit happens here (not at upload-init) so a slow client
    # can't sit on a reservation that prevents others from uploading.
    await enforce_chapter_quota(user, db, cfg.rate_limit, auth, count=1)

    if not body.parts:
        raise HTTPException(400, "Empty parts list")

    raw_number = (
        (body.number or "").strip()
        or await _next_sequential_number(db, project_id)
    )

    pj = Projects(db, paths, stores)
    handle = InboxHandle(
        chapter_id=0,                   # filled in by queue_chapter
        tmp_id=body.tmp_id,
        upload_id=body.upload_id,
        parts=tuple(
            CompletedPart(number=p.number, etag=p.etag) for p in body.parts
        ),
        title=body.title,
    )

    try:
        chapter_id = await pj.queue_chapter(
            project_id, raw_number, handle, title=body.title,
        )
    except Exception as e:
        logger.exception(
            "queue_chapter failed (project=%s number=%s tmp=%s)",
            project_id, raw_number, body.tmp_id,
        )
        raise HTTPException(500, f"queue failed: {e}") from e

    await record_chapter_consume(user, db, auth, chapter_id, project_id)

    data = await db.get_chapter_with_status(chapter_id, project_id)
    if data is None:
        raise HTTPException(500, "Chapter created but lookup failed")
    return chapter_out(data)


@router.post(
    "/{project_id}/chapters/upload-abort",
    status_code=204,
)
async def upload_abort(
    project_id: int,
    body:  UploadAbortBody,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    inbox: ChapterInbox  = Depends(get_inbox),
):
    """Drop a half-finished multipart upload. Idempotent."""
    await require_project_owner(project_id, user, db)
    try:
        await inbox.abort_multipart(
            tmp_id=body.tmp_id, upload_id=body.upload_id,
        )
    except Exception as e:
        logger.warning(
            "abort_multipart failed (tmp=%s upload=%s): %s",
            body.tmp_id, body.upload_id, e,
        )


# ── Local inbox PUT route (dev only) ──────────────────────────────
#
# When `storage.inbox.type == 'local'`, `LocalInbox.create_multipart`
# returns `${public_api_url}/api/_inbox/...` URLs. This sibling
# router accepts those PUTs so the SDK code path is identical in dev
# and prod. In prod (`s3`), this route never receives traffic — the
# presigned URLs point at the storage provider directly.

local_router = APIRouter(prefix="/api/_inbox", tags=["upload-local"])


@local_router.put("/{tmp_id}/{upload_id}/{number}")
async def local_inbox_put(
    tmp_id:    str,
    upload_id: str,
    number:    int,
    request:   Request,
    inbox:     ChapterInbox = Depends(get_inbox),
):
    """Accept a PUT into the local-dev inbox. Returns ETag in headers
    so the SDK can read it the same way it reads it from S3.

    Auth is intentionally unrestricted: the URL itself is the capability
    (only the user who got it back from `upload-init` knows tmp_id +
    upload_id), matching the security model of an S3 presigned URL.
    Production should use `storage.inbox.type=s3` so this route is
    never reached.
    """
    if not isinstance(inbox, LocalInbox):
        raise HTTPException(404, "Local inbox is not configured")

    try:
        target = inbox.part_path(tmp_id, upload_id, number)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    target.write_bytes(body)

    # ETag scheme matches LocalInbox.complete_multipart's verifier:
    # sha256 hex digest of the part bytes.
    import hashlib
    etag = hashlib.sha256(body).hexdigest()
    return _PutResponse(etag)


def _PutResponse(etag: str):  # noqa: N802 — small wrapper, lowercased makes routing read odd
    from fastapi import Response
    r = Response(status_code=200)
    r.headers["ETag"] = etag
    # Browser CORS: must be in Access-Control-Expose-Headers so JS can
    # read it. The middleware already sets a permissive list.
    return r


# ── Helpers ───────────────────────────────────────────────────────


_NUMBER_RE = re.compile(r"(?:^|[^\d])(\d+(?:\.\d+)?)")


async def _next_sequential_number(db: Store, project_id: int) -> str:
    """Return `floor(max numeric number) + 1`, or "1" when the project
    is empty or contains only label-only chapters (Extra/Oneshot).

    Last-resort default for uploads where the client did not supply a
    `number` — better to land at "1", "2", "3"… than to bounce a valid
    pile of pages with a 400.
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
