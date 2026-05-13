"""Chapter ingestion — browser-direct multipart.

Supports ext / upload AND source-backed materials.
Source-backed materials (HappyMH, MangaDex, OTruyen) normally pull
chapters from the manifest at read time, but when the client pre-downloads
pages (CORS / auth workarounds) and uploads a ZIP, this route handles it.

Flow:

  1. client → POST /material/{id}/chapter/upload-init { byte_size }
     server → ownership gate (ext/upload: imported_by = user)
            → quota check
            → inbox.create_multipart → presigned PUT URLs
     server → returns { tmp_id, upload_id, parts, part_size, expires_in }

  2. client → PUT each part to its presigned URL

  3. client → POST /material/{id}/chapter/upload-finalize {
                  tmp_id, upload_id, parts, number?, label?,
                  upstream_url?
              }
     server → dedup existing manifest chapter (upstream_url) or create new
            → persists inbox handle
            → enqueues prepare task
            → returns chapter id

  X. /upload-abort drops a half-finished multipart upload. Idempotent.

The actual zip download / unpack / prepare / bunle.pack happens in
the prepare worker (`workers.loop._handle_prepare`). API hot path
stays under ~100ms regardless of chapter size.

LocalInbox dev backend simulates multipart with a sibling
`PUT /api/_inbox/{tmp_id}/{upload_id}/{number}` route so the client
SDK has one code path in dev and prod.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from typoon.adapters.inbox import (
    DEFAULT_PART_SIZE, ChapterInbox, CompletedPart, InboxHandle, LocalInbox,
)
from typoon.api.deps import (
    get_auth_cfg, get_config, get_inbox, get_store, require_user,
)
from typoon.api.models import ChapterOut, ChapterTranslationOverlay
from typoon.api.quota import enforce_chapter_quota, record_consume
from typoon.api.routes._shared import require_material, require_material_admin
from typoon.config import AuthConfig, Config
from typoon.storage import Store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/material", tags=["upload"],
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

# Browser-direct PUTs always flow through the public origin's `/r2`
# URL Mapping. In production the public origin IS the DA origin
# (`https://<app_id>.discordsays.com`) — every public path is fronted
# by the Discord proxy regardless of whether the client is the DA
# iframe or a plain browser. In dev `public_base_url` points at the
# local API directly, the LocalInbox URLs are not R2 hosts, and this
# rewrite is a no-op.
#
# SigV4 signs Host only; the mapping target is the original R2 host,
# so the proxy forwards with the right Host and the signature stays
# valid regardless of which origin made the PUT.
_R2_HOST_RE = re.compile(r"^[a-z0-9-]+\.r2\.cloudflarestorage\.com$", re.I)


def _rewrite_presigned_for_public(url: str, public_origin: str) -> str:
    """Rewrite an R2 presigned URL to flow through `<public_origin>/r2/`.

    Empty origin or non-R2 URLs (LocalInbox pointed at the API itself)
    pass through unchanged.
    """
    from urllib.parse import urlsplit, urlunsplit

    if not public_origin:
        return url
    parts = urlsplit(url)
    if not parts.hostname or not _R2_HOST_RE.match(parts.hostname):
        return url
    base = urlsplit(public_origin)
    return urlunsplit(
        (base.scheme, base.netloc, f"/r2{parts.path}", parts.query, ""),
    )


async def _gate_for_chapter_upload(
    material_id: int, user: dict, db: Store,
) -> dict:
    """Allow chapter uploads for ext / upload materials (owner-gated) and
    source-backed materials (any authenticated user, since source pages
    are fetched client-side before upload due to CORS / auth constraints)."""
    mat = await require_material(material_id, db)
    if mat["origin"] in ("extension", "upload"):
        # ext/upload: only the importer may add chapters.
        if mat.get("imported_by") != user["id"]:
            raise HTTPException(403, "Only the importer can add chapters")
    elif mat["origin"] == "source":
        # Source-backed: any logged-in user may upload pre-downloaded pages.
        pass
    else:
        raise HTTPException(400, f"Unsupported material origin: {mat['origin']!r}")
    return mat


# ── Schemas ───────────────────────────────────────────────────────────


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
    tmp_id:       str = Field(min_length=8, max_length=64)
    upload_id:    str = Field(min_length=1, max_length=512)
    parts:        list[FinalizePart] = Field(min_length=1, max_length=_MAX_PARTS)
    label:        str | None = None
    upstream_url: str | None = None
    # Manifest-supplied normalised chapter key. The only chapter
    # identity field — display strings live on `label`. For ext /
    # upload origins without a manifest spec, the client may compute
    # a stable slug client-side; an isolated upload still gets its
    # own work_chapter row that community vote can merge later.
    number_norm:  str | None = None


class UploadAbortBody(BaseModel):
    tmp_id:    str = Field(min_length=8, max_length=64)
    upload_id: str = Field(min_length=1, max_length=512)


# ── Routes ────────────────────────────────────────────────────────────


@router.post(
    "/{material_id}/chapter/upload-init",
    response_model=UploadInitOut,
)
async def upload_init(
    material_id: int,
    body:        UploadInitBody,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    cfg:   Config        = Depends(get_config),
    auth:  AuthConfig    = Depends(get_auth_cfg),
    inbox: ChapterInbox  = Depends(get_inbox),
):
    """Begin a multipart upload for one chapter zip.

    Quota is enforced **before** any presigned URL leaves the server,
    so a maxed-out user gets 429 instantly instead of after PUTting
    100 MB of data. Race window between init and finalize closes by
    re-checking + consuming at finalize time.
    """
    await _gate_for_chapter_upload(material_id, user, db)
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

    base_origin = cfg.server.public_base_url
    parts_out = [
        InitPart(
            number=p.number,
            url=_rewrite_presigned_for_public(p.url, base_origin),
        )
        for p in urls
    ]

    return UploadInitOut(
        tmp_id=tmp_id,
        upload_id=upload_id,
        parts=parts_out,
        part_size=DEFAULT_PART_SIZE,
        expires_in=3600,  # PRESIGN_TTL_SECONDS — keep clients in sync
    )


@router.post(
    "/{material_id}/chapter/upload-finalize",
    response_model=ChapterOut,
    status_code=202,
)
async def upload_finalize(
    material_id: int,
    body:        UploadFinalizeBody,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    cfg:   Config        = Depends(get_config),
    auth:  AuthConfig    = Depends(get_auth_cfg),
):
    """Create the chapter row, persist the inbox handle, enqueue
    prepare. The actual heavy lifting (complete_multipart, fetch zip,
    unpack, prepare, pack prepared.bnl) is deferred to the prepare
    worker so the HTTP round-trip stays short (~50ms).
    """
    await _gate_for_chapter_upload(material_id, user, db)

    # Re-check the quota gate at finalize time so a slow client can't
    # sit on a reservation while others are blocked.
    await enforce_chapter_quota(user, db, cfg.rate_limit, auth, count=1)

    if not body.parts:
        raise HTTPException(400, "Empty parts list")

    # Dedup: if a chapter with this upstream_url already exists for the
    # material (e.g. another user uploaded the same source chapter, or
    # a manifest-coordinated translate created the row first), reuse it
    # so we don't fork the chapter into duplicate rows. The prepare
    # worker's CAS dedup on prepared_hash handles content-level dedup
    # too, but row-level dedup keeps the cache key (chapter_id, ...)
    # consistent across users.
    chapter_id: int | None = None
    if body.upstream_url:
        existing = await db.find_chapter_by_upstream(
            material_id, body.upstream_url,
        )
        if existing is not None:
            chapter_id = existing["id"]

    if chapter_id is None:
        # `number_norm` is the canonical identity. Caller (SPA / SDK)
        # computes it via the manifest's declarative normaliser.
        # When absent (ext / upload origin with no manifest spec),
        # generate a sequential fallback so an upload still lands.
        number_norm = (body.number_norm or "").strip() or await _next_sequential_number(
            db, material_id,
        )
        chapter_id = await db.create_chapter(
            material_id,
            number_norm=number_norm,
            label=body.label,
            upstream_url=body.upstream_url,
        )

    # Persist inbox handle so the prepare worker can complete the
    # multipart upload + fetch the zip.
    handle = InboxHandle(
        chapter_id=chapter_id,
        tmp_id=body.tmp_id,
        upload_id=body.upload_id,
        parts=tuple(
            CompletedPart(number=p.number, etag=p.etag) for p in body.parts
        ),
        title=body.label,
    )
    await db.set_inbox_handle(handle)

    # Enqueue prepare on this chapter. Worker fans out from there.
    await db.enqueue_task(
        target_kind="chapter", target_id=chapter_id, stage="prepare",
    )

    # Quota commit AFTER the chapter row exists — the consume row is
    # tied to a translation_id in the new schema, but at upload time
    # we don't have one yet. We hold the consume until the first
    # translate spawn happens (POST /api/translate). For symmetry the
    # legacy flow recorded at upload-finalize; in the new model the
    # cost belongs to translate, not ingest. We deliberately skip the
    # record here.

    # Return a ChapterOut with empty translation overlay (none yet).
    chapter = await db.get_chapter(chapter_id)
    if chapter is None:
        raise HTTPException(500, "Chapter created but lookup failed")
    return ChapterOut(
        id=chapter["id"],
        material_id=chapter["material_id"],
        position=chapter["position"],
        number=chapter["number_norm"],
        label=chapter.get("label"),
        upstream_url=chapter.get("upstream_url"),
        page_count=int(chapter.get("page_count") or 0),
        updated_at=chapter.get("updated_at"),
        translations=[],
    )


@router.post(
    "/{material_id}/chapter/upload-abort",
    status_code=204,
)
async def upload_abort(
    material_id: int,
    body:        UploadAbortBody,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    inbox: ChapterInbox  = Depends(get_inbox),
):
    """Drop a half-finished multipart upload. Idempotent."""
    await _gate_for_chapter_upload(material_id, user, db)
    try:
        await inbox.abort_multipart(
            tmp_id=body.tmp_id, upload_id=body.upload_id,
        )
    except Exception as e:
        logger.warning(
            "abort_multipart failed (tmp=%s upload=%s): %s",
            body.tmp_id, body.upload_id, e,
        )


# ── Local inbox PUT route (dev only) ─────────────────────────────────
#
# When `storage.inbox.type == 'local'`, `LocalInbox.create_multipart`
# returns `${public_base_url}/api/_inbox/...` URLs. This sibling router
# accepts those PUTs so the SDK code path is identical in dev and prod.
# In prod (`s3`), this route never receives traffic — presigned URLs
# point at the storage provider directly.

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
    so the SDK reads it the same way it reads it from S3.

    Auth is intentionally unrestricted: the URL itself is the capability
    (only the user who got it back from `upload-init` knows tmp_id +
    upload_id), matching the security model of an S3 presigned URL.
    Production uses `storage.inbox.type=s3` so this route is never hit.
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
    etag = hashlib.sha256(body).hexdigest()
    r = Response(status_code=200)
    r.headers["ETag"] = etag
    # Browser CORS: must be in Access-Control-Expose-Headers so JS can
    # read it. The middleware already sets a permissive list.
    return r


# ── Helpers ──────────────────────────────────────────────────────────


_NUMBER_RE = re.compile(r"(?:^|[^\d])(\d+(?:\.\d+)?)")


async def _next_sequential_number(db: Store, material_id: int) -> str:
    """Return `floor(max numeric number_norm) + 1`, or "1" when the
    material is empty or contains only label-only chapters (Extra/
    Oneshot).

    Last-resort default when the client (manifest plugin or upload
    UI) didn't supply a number_norm — better to land at "1", "2", "3"
    than to bounce a valid pile of pages with a 400.
    """
    rows = await db.list_chapters(material_id)
    max_num = 0.0
    for r in rows:
        try:
            n = float(r["number_norm"])
        except (TypeError, ValueError):
            continue
        if n > max_num:
            max_num = n
    return str(math.floor(max_num) + 1)
