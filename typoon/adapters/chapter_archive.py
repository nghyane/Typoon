"""Chapter archive helpers — deterministic Bunle keys.

Three archive types per chapter:

- prepared.bnl   server-side only (read by scan/translate workers); key
                 carries the raw (project, chapter) identity since no
                 browser ever fetches it.
- masks.npz      server-side only (read by render worker); same.
- render.bnl     PUBLIC — fetched by the browser via CDN. Key is a
                 deterministic HMAC token of (project, chapter) so the
                 URL doesn't expose internal IDs.

The token is derived once from a chapter pair using a fixed salt; same
inputs always produce the same token, so cache keys at the CDN edge are
stable across re-renders. To bust the cache when content changes, the
caller appends a `?v=<updated_at>` query string to the public URL.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path

import bunle

from typoon.adapters.blob_store import BlobStore


def archive_token(project_id: int, chapter_id: int, salt: bytes) -> str:
    """16-char base64url HMAC token derived from (project, chapter, salt).

    Deterministic, unguessable (~96 bits entropy), one-way. Same chapter
    always maps to the same token → CDN cache key is stable. Rotating
    the salt invalidates every URL — used as a manual nuke, not as
    routine cache control.
    """
    msg = f"{project_id}:{chapter_id}".encode()
    digest = hmac.new(salt, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()[:16]


def archive_dir_prefix(project_id: int, chapter_id: int) -> str:
    return f"p/{project_id}/c/{chapter_id}"


def prepared_key(project_id: int, chapter_id: int) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/prepared.bnl"


def masks_key(project_id: int, chapter_id: int) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/masks.npz"


def render_key(project_id: int, chapter_id: int, salt: bytes) -> str:
    """Public render archive key. Path uses the HMAC token so the URL
    served to browsers does not embed (project_id, chapter_id)."""
    return f"render/{archive_token(project_id, chapter_id, salt)}.bnl"


async def pack_and_upload(
    *,
    src_dir: Path,
    archive_path: Path,
    key: str,
    store: BlobStore,
) -> tuple[int, str]:
    """Pack `src_dir` into a Bunle archive at `archive_path`, validate, upload.

    Returns (page_count, locator) — the locator string is what the
    backend wrote (path for path-based stores, opaque id for ones like
    Drive). Caller persists it on the chapter row alongside
    `store.backend_name` so reads dispatch back to the right backend.
    """
    bunle.pack_dir(str(src_dir), str(archive_path))
    bunle.validate(str(archive_path))
    info = bunle.info(str(archive_path))
    locator = await store.put(key, archive_path)
    return info["page_count"], locator
