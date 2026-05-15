"""Material routes.

A "material" is one source's manifestation of a manga — (source,
upstream_ref) for plugin-backed rows, per-row for ext / upload. The
user-facing manga page lives at `GET /work/{id}` (Work-centric); the
material routes here exist only for create / edit / delete of the
underlying source-bound row. Resolving the manga UI from a material
goes through `materials.work_id`.

Endpoints:
    POST   /api/material/import          source-backed lookup-or-create
    POST   /api/material                  ext / upload creation
    PATCH  /api/material/{id}             editable fields (ext/upload owner only)
    DELETE /api/material/{id}             cascade (ext/upload owner only)
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_config, get_storage, get_store, require_user
from typoon.api.models import MaterialOut
from typoon.api.routes._shared import require_material_admin
from typoon.storage import Store

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/material", tags=["material"],
    dependencies=[Depends(require_user)],
)


# ── Request bodies ────────────────────────────────────────────────────


class ImportBody(BaseModel):
    """Idempotent import for source-backed materials.

    The manifest runtime on the SPA already fetched the upstream page
    and resolved the display snapshot; we trust it for first-write but
    refresh on later imports via a background job (out of scope here).
    """
    source:        str
    upstream_ref:  str

    title:         str
    cover_url:     str | None = None
    description:   str | None = None
    author:        str | None = None
    status:        str | None = None
    languages:     list[str] = []
    title_native:  str | None = None
    title_alt:     list[str] = []
    cross_refs:    dict | None = None
    nsfw:          bool = False


class CreateLocalBody(BaseModel):
    origin:       str        # 'extension' | 'upload'
    title:        str
    cover_url:    str | None = None
    description:  str | None = None
    author:       str | None = None
    nsfw:         bool = False


class PatchMaterialBody(BaseModel):
    title:        str | None = None
    cover_url:    str | None = None
    description:  str | None = None
    nsfw:         bool | None = None


# ── Import / create ───────────────────────────────────────────────────


@router.post("/import", response_model=MaterialOut)
async def import_material(
    body: ImportBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Idempotent for (source, upstream_ref). A second call returns the
    same row id — `imported_by` is set on first insert only."""
    material_id = await db.get_or_create_source_material(
        source=body.source,
        upstream_ref=body.upstream_ref,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        author=body.author,
        status=body.status,
        languages=body.languages,
        title_native=body.title_native,
        title_alt=body.title_alt,
        cross_refs=body.cross_refs,
        nsfw=body.nsfw,
        imported_by=user["id"],
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


@router.post("", response_model=MaterialOut)
async def create_local_material(
    body: CreateLocalBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Per-row material for extension capture or user upload. No dedup."""
    if body.origin not in ("extension", "upload"):
        raise HTTPException(
            400, "origin must be 'extension' or 'upload'",
        )
    material_id = await db.create_local_material(
        origin=body.origin,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        author=body.author,
        nsfw=body.nsfw,
        imported_by=user["id"],
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


# ── Read ──────────────────────────────────────────────────────────────
#
# `GET /material/{id}` was removed in the schema 23 cutover; the canonical
# detail page lives at `GET /work/{id}` now (Step 2 of the Work refactor
# adds it). External callers should resolve material → work via the
# import response or the work payload.


# ── Edit ──────────────────────────────────────────────────────────────


@router.patch("/{material_id}", response_model=MaterialOut)
async def patch_material(
    material_id: int,
    body:        PatchMaterialBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Ext / upload owner can update display fields. Source-backed
    materials are immutable (their state mirrors the manifest)."""
    await require_material_admin(material_id, user, db)
    await db.update_material_metadata(
        material_id,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        nsfw=body.nsfw,
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


# ── Cover upload ──────────────────────────────────────────────────────
#
# User-supplied cover image. Distinct from `cover_url` paste in the
# PATCH body above — that one trusts a remote URL the client gives us;
# this one accepts raw bytes from the user's device, stores them in
# our own ArtifactStore, and writes the resulting public URL back into
# `materials.cover_url`. From the frontend's POV both paths produce
# the same row shape — only the source of the bytes differs.
#
# Scope (today): ext / upload origin only. Source-backed materials
# are immutable (their state mirrors the manifest snapshot).
#
# Validation:
#   • mime: jpeg / png / webp only — no SVG (XSS risk), no GIF
#     (animated covers add no value), no HEIC (browser support
#     uneven).
#   • size: ≤ 2 MiB declared and ≤ 2 MiB actually read.
#   • decode: must be a real image Pillow can open. Catches files
#     renamed to .jpg but actually HTML / archives / etc.
#
# Storage layout:
#   covers/m<material_id>/cover.<ext>
#
# Re-uploading overwrites the same key so the served URL stays stable
# across edits; the cache-bust query string (`?v=<updated_at>`) at the
# Cover component does the cold-revalidate.

# Mime → file extension. Restrictive on purpose — the value lands in
# a path served from our origin, so allowing exotic types only adds
# attack surface.
_ALLOWED_COVER_MIMES = {
    "image/jpeg": "jpg",
    "image/png":  "png",
    "image/webp": "webp",
}

# Server-side hard cap. The browser SDK should reject earlier so the
# user sees a clear error before the request hits the wire, but we
# enforce here too in case a non-SDK client tries to bypass it.
_MAX_COVER_BYTES = 2 * 1024 * 1024


@router.post("/{material_id}/cover", response_model=MaterialOut)
async def upload_cover(
    material_id: int,
    file:   UploadFile           = File(...),
    user:   dict                 = Depends(require_user),
    db:     Store                = Depends(get_store),
    stores: StorageRegistry      = Depends(get_storage),
):
    """Upload a cover image for an ext / upload material.

    Returns the refreshed MaterialOut so the client can re-render
    without a separate refetch — `cover_url` will point at the newly
    stored asset.
    """
    await require_material_admin(material_id, user, db)

    # Validate mime BEFORE reading the body so a malicious client
    # can't make us buffer 2 MiB just to reject it on extension. The
    # body read below still caps in case content_type lies; this
    # check just shortens the common-case rejection path.
    content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    ext = _ALLOWED_COVER_MIMES.get(content_type)
    if ext is None:
        raise HTTPException(
            415,
            f"Unsupported cover type {content_type!r}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COVER_MIMES))}.",
        )

    # Read with a hard byte cap. `UploadFile.read(n)` returns at most
    # `n` bytes; we read `_MAX_COVER_BYTES + 1` and reject if anything
    # exceeds the cap, so a 1 GiB upload aborts at 2 MiB + 1 instead
    # of filling RAM.
    data = await file.read(_MAX_COVER_BYTES + 1)
    if len(data) > _MAX_COVER_BYTES:
        raise HTTPException(
            413,
            f"Cover too large (> {_MAX_COVER_BYTES // 1024} KiB).",
        )
    if not data:
        raise HTTPException(400, "Empty cover upload.")

    # Decode to confirm it's a real image and not e.g. an HTML page
    # renamed to .jpg. Decoding once also normalises the byte layout
    # (e.g. stripping EXIF that could carry geolocation) on the way
    # to the store, since `Image.save` re-encodes from the decoded
    # pixels. We keep the same format the user uploaded so the
    # extension on disk matches the bytes — no transcoding surprises.
    try:
        from PIL import Image, UnidentifiedImageError

        img = Image.open(io.BytesIO(data))
        img.load()                       # force full decode
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"Cover is not a readable image: {e}") from e

    # Re-encode through Pillow to strip EXIF + verify integrity.
    # We use the SAME format the request claimed so the file extension
    # stays truthful. Pillow's format detection is independent of the
    # request mime; we trust the request mime for the EXTENSION
    # because we already gated on the allow-list above.
    buf = io.BytesIO()
    save_kwargs: dict = {}
    if ext == "jpg":
        save_format = "JPEG"
        save_kwargs["quality"] = 90
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")     # JPEG can't carry alpha
    elif ext == "png":
        save_format = "PNG"
    else:
        save_format = "WEBP"
        save_kwargs["quality"] = 90
    img.save(buf, format=save_format, **save_kwargs)
    encoded = buf.getvalue()

    # Stream the encoded bytes through ArtifactStore. `put` takes a
    # filesystem path (matches the existing chapter archive flow), so
    # we materialise to a temp file. Tempfile lives on the API node,
    # not the worker, so the deletion-on-close suffices — no cross-
    # host cleanup needed.
    locator_key = f"covers/m{material_id}/cover.{ext}"
    with tempfile.NamedTemporaryFile(
        prefix=f"cover-m{material_id}-", suffix=f".{ext}", delete=False,
    ) as tmp:
        tmp.write(encoded)
        tmp_path = Path(tmp.name)
    try:
        stored_locator = await stores.public.put(locator_key, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # The URL we persist is what the browser will load. ArtifactStore
    # backends differ (LocalArtifactStore returns `/files/...`,
    # HuggingFaceArtifactStore returns `https://<cdn>/...`), and the
    # frontend's `coverUrl()` already handles both shapes — relative
    # paths go through `api.base`, absolute URLs go through the CDN
    # proxy. Bust the cache via the same `version` flow the rest of
    # the Cover component uses; persisting the locator-derived URL
    # keeps the cache-bust client-side (the row's `updated_at` is the
    # version key).
    public_url = stores.public.url(stored_locator)

    await db.update_material_metadata(material_id, cover_url=public_url)

    refreshed = await db.get_material(material_id)
    assert refreshed is not None
    return MaterialOut.from_row(refreshed)



#
# Auto-enrich is client-driven: the SPA fans search across link
# plugins (Anilist, MAL, …) when it opens a Work whose cross_refs
# are empty, and POSTs whatever it found back here. This endpoint
# MERGES the suggested cross_refs additively — never overwrites a
# value that already exists. Anyone authenticated may submit; the
# operation is idempotent.
#
# After the merge we ask the linker to re-evaluate whether this
# material's Work should attach to an existing Work that shares a
# cross_ref namespace. The linker decides; this route doesn't merge
# anything itself.


class EnrichMetadataBody(BaseModel):
    """Client-supplied metadata fields to merge into the material.

    Every field is optional; the storage layer fills only the columns
    it has data for and never overwrites manifest-authoritative values.

      cross_refs   — `{namespace: id}` (anilist, mal, mdex_uuid, …).
                     Existing namespaces win on conflict.
      title_native — fills only when currently null.
      title_alt    — set-union with existing list (dedupes).
      title_locale — BCP-47 → display string. Existing langs win.
      start_year   — fills only when currently null.
      description  — fills only when currently null.
      cover_url    — fills only when currently null. Lets ext/upload
                     materials (created without a cover) pick up the
                     canonical artwork from MangaBaka / MangaDex on
                     the first enrich pass; source-backed materials
                     keep their per-scanlator cover untouched.

    `source_signals` is reserved for an audit log (which plugin
    claimed what with what confidence). Stored nowhere yet; we keep
    the parameter so the client signature is stable.
    """
    cross_refs:    dict[str, str | int | float] | None = None
    title_native:  str | None = None
    title_alt:     list[str] | None = None
    title_locale:  dict[str, str] | None = None
    start_year:    int | None = None
    description:   str | None = None
    cover_url:     str | None = None
    source_signals: list[dict] | None = None


@router.post(
    "/{material_id}/enrich-metadata",
    response_model=MaterialOut,
)
async def enrich_material_metadata(
    material_id: int,
    body:        EnrichMetadataBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Merge client-found metadata onto a material additively.

    The SPA's link-plugin runtime calls this after fanning search out
    across services (Anilist, MangaDex, …). We accept whatever it
    found; the storage layer enforces "existing-wins on conflict" so
    manifest data is never overwritten by enriched values.
    """
    mat = await db.get_material(material_id)
    if mat is None:
        raise HTTPException(404, "Material not found")

    _ = body.source_signals     # audit log — not implemented yet
    _ = user

    await db.merge_material_metadata(
        material_id,
        cross_refs=   body.cross_refs,
        title_native= body.title_native,
        title_alt=    body.title_alt,
        title_locale= body.title_locale,
        start_year=   body.start_year,
        description=  body.description,
        cover_url=    body.cover_url,
    )
    refreshed = await db.get_material(material_id)
    assert refreshed is not None
    return MaterialOut.from_row(refreshed)


# ── Delete ────────────────────────────────────────────────────────────


@router.delete("/{material_id}", status_code=204)
async def delete_material(
    material_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    """Ext / upload owner can delete; source-backed materials are
    immutable to users (DMCA admin path handles those).

    Cascade drops chapters → drafts → translations → bubbles etc. via
    FK ON DELETE CASCADE. We strip pipeline blobs (prepared, masks)
    and any render archives before the DB cascade, otherwise the
    locator pointers vanish first and we orphan blobs on remote
    storage.

    Cover assets are best-effort cleaned too — they live on the public
    store at a deterministic key, so we issue one delete per allowed
    format without needing to track which one was actually written.
    """
    from typoon.adapters.chapter_archive import masks_key, prepared_key

    await require_material_admin(material_id, user, db)
    cfg = get_config()
    salt = cfg.storage.archive_path_salt.encode()
    chapters = await db.list_chapters(material_id)
    for ch in chapters:
        await stores.pipeline.delete(prepared_key(ch["id"]))
        await stores.pipeline.delete(masks_key(ch["id"]))
        # Per-draft default renders + per-translation overrides each
        # live on whichever backend wrote them. Dispatch by backend so
        # multi-backend coexistence (e.g. mid-migration) doesn't leak
        # blobs.
        await _drop_archives_for_chapter(db, stores, ch["id"], salt=salt)

    # User-uploaded cover lives at `covers/m<id>/cover.<ext>` on the
    # public store. We don't persist which extension was written, so
    # we issue one best-effort delete per allowed format. `delete` is
    # idempotent and returns False for missing keys, so the two
    # non-matching extensions are cheap no-ops on local; on HF they
    # cost one API round-trip each but only on material deletion (rare
    # vs. read traffic), so the simplicity wins over carrying a
    # cover_ext column.
    await _drop_cover(stores, material_id)

    await db.delete_material(material_id)


async def _drop_cover(stores: StorageRegistry, material_id: int) -> None:
    """Best-effort delete of the cover blob, swallowing per-format
    failures. A storage outage here must not block the DB cascade —
    a stale cover file is a cheap leak compared to leaving an
    undeletable material row."""
    for ext in _ALLOWED_COVER_MIMES.values():
        key = f"covers/m{material_id}/cover.{ext}"
        try:
            await stores.public.delete(key)
        except Exception as e:                            # noqa: BLE001
            logger.warning(
                "cover cleanup: delete %s failed: %s", key, e,
            )


# ── Cleanup helpers ──────────────────────────────────────────────────


async def _drop_archives_for_chapter(
    db:      Store,
    stores:  StorageRegistry,
    chapter_id: int,
    *,
    salt:    bytes,
) -> None:
    """Remove every render archive for a chapter — draft defaults and
    per-translation overrides — across whichever backends wrote them.

    Failures are logged but not raised: an unreachable backend or
    already-gone blob must not block the DB cascade. The pipeline
    blobs (prepared/masks) are handled by the caller because their
    backend (always `stores.pipeline`) is fixed.
    """
    from typoon.adapters.chapter_archive import render_key

    drafts = await db.list_drafts_for_chapter(chapter_id)
    for d in drafts:
        await _drop_archive(
            stores,
            backend=d.get("archive_backend"),
            locator=d.get("archive_locator")
                    or render_key("draft", d["id"], salt),
            label=f"draft={d['id']}",
        )

    translations = await db.list_all_translations_for_chapter(chapter_id)
    for t in translations:
        await _drop_archive(
            stores,
            backend=t.get("archive_backend"),
            locator=t.get("archive_locator")
                    or render_key("translation", t["id"], salt),
            label=f"translation={t['id']}",
        )


async def _drop_archive(
    stores:  StorageRegistry,
    *,
    backend: str | None,
    locator: str | None,
    label:   str,
) -> None:
    """Best-effort delete of a single archive blob. Used by chapter /
    material cleanup; tolerates missing backend or stale locator."""
    if not backend or not locator:
        return
    try:
        reader = stores.reader(backend)
    except RuntimeError:
        logger.warning(
            "skip %s archive cleanup: backend %r not configured",
            label, backend,
        )
        return
    try:
        await reader.delete(locator)
    except Exception:
        logger.exception(
            "archive cleanup failed for %s (backend=%s locator=%s)",
            label, backend, locator,
        )
