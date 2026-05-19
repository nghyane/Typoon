"""Chapter archive helpers — deterministic Bunle keys for the v5
material architecture.

Three archive types in the pipeline:

- prepared.bnl   chapter-scoped (Layer 1). Built once per chapter
                 (CAS-deduped via chapter.prepared_hash). Read by
                 scan + translate workers; never browser-fetched.

- masks.npz      chapter-scoped (Layer 1). Built once per chapter
                 by the scan stage; read by the render worker.

- render.bnl     translation or draft scoped. PUBLIC — browser
                 fetches via DA proxy → bunle CDN. Key is an
                 HMAC token derived from (target_kind, target_id, salt)
                 so the URL does not leak internal numeric ids.

Token rotation: bumping BLOB_SALT invalidates every cached render URL.
Used as a nuke (DMCA mass clear, key compromise) — clients re-resolve
via the API on next read and get fresh tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import Literal

import bunle

from typoon.adapters.blob_store import BlobStore


TargetKind = Literal["chapter", "draft", "translation"]


def archive_token(
    target_kind: TargetKind, target_id: int, salt: bytes,
) -> str:
    """16-char base64url HMAC token derived from (target_kind, target_id, salt).

    Deterministic, unguessable (~96 bits entropy), one-way. Same target
    always maps to the same token → CDN cache key is stable. Rotating
    the salt invalidates every URL — used as a manual nuke.
    """
    msg = f"{target_kind}:{target_id}".encode()
    digest = hmac.new(salt, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()[:16]


def chapter_dir_prefix(chapter_id: int) -> str:
    return f"c/{chapter_id}"


def prepared_key(chapter_id: int) -> str:
    """Pipeline blob key for prepared.bnl. Chapter-scoped — shared
    across every translation of this chapter."""
    return f"{chapter_dir_prefix(chapter_id)}/prepared.bnl"


def masks_key(chapter_id: int) -> str:
    """Pipeline blob key for masks.npz. Chapter-scoped."""
    return f"{chapter_dir_prefix(chapter_id)}/masks.npz"


def render_key(
    target_kind: TargetKind, target_id: int, salt: bytes,
) -> str:
    """Public render archive key. Path uses the HMAC token so the URL
    served to browsers does not embed internal ids.

    `target_kind`:
      - 'draft'       — default render shared by every translation
                        referencing this draft with no edits
      - 'translation' — per-user render when sparse edits diverge
    """
    if target_kind == "chapter":
        raise ValueError(
            "chapter-scoped render not supported — render is per draft "
            "or per translation"
        )
    return f"render/{archive_token(target_kind, target_id, salt)}.bnl"


async def pack_and_upload(
    *,
    src_dir: Path,
    archive_path: Path,
    key: str,
    store: BlobStore,
) -> tuple[int, str]:
    """Pack `src_dir` into a Bunle archive, validate, upload.

    Returns (page_count, locator) — locator is what the backend wrote
    (path for path-based stores, opaque id for ones like Drive).
    Caller persists it on the chapter / translation row alongside
    `store.backend_name` so reads dispatch back to the right backend.
    """
    bunle.pack_dir(str(src_dir), str(archive_path))
    bunle.validate(str(archive_path))
    info = bunle.info(str(archive_path))
    locator = await store.put(key, archive_path)
    return info["page_count"], locator
