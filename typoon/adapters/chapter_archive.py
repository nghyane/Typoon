"""Chapter archive helpers — content-addressed Bunle keys, sha256 revs.

Owns the `prepared-<rev>.bnl` / `render-<rev>.bnl` key shape so stages and
the API agree on names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import bunle

from typoon.adapters.artifact_store import ArtifactStore


@dataclass(frozen=True)
class ArchiveResult:
    key: str
    rev: str
    page_count: int


def archive_dir_prefix(project_id: int, chapter_id: int) -> str:
    return f"p/{project_id}/c/{chapter_id}"


def prepared_key(project_id: int, chapter_id: int, rev: str) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/prepared-{rev}.bnl"


def render_key(project_id: int, chapter_id: int, rev: str) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/render-{rev}.bnl"


def archive_rev(path: Path) -> str:
    """sha256(file)[:8] — used as the content-addressed key suffix."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


async def pack_and_upload(
    *,
    src_dir: Path,
    archive_path: Path,
    key_builder,
    store: ArtifactStore,
) -> ArchiveResult:
    """Pack a directory of pre-encoded pages, validate, upload, return key.

    `key_builder(rev: str) -> str` builds the storage key from the computed
    revision. Caller is responsible for DB pointer flip and old-key delete.
    """
    bunle.pack_dir(str(src_dir), str(archive_path))
    bunle.validate(str(archive_path))
    info = bunle.info(str(archive_path))
    rev = archive_rev(archive_path)
    key = key_builder(rev)
    await store.put_file(key, archive_path)
    return ArchiveResult(key=key, rev=rev, page_count=info["page_count"])
