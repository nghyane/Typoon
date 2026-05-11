"""Prepare a chapter and persist it as a Bunle archive.

`stages.prepare.prepare_chapter` writes lossless WebP pages into a temp
directory; this orchestrator packs them into `prepared.bnl` and uploads
to the chapter's prepared key. The DB pointer flip
(`Store.set_prepared_done`) is the caller's responsibility.
"""

from __future__ import annotations

from pathlib import Path

from typoon.adapters.blob_store import BlobStore
from typoon.adapters.chapter_archive import pack_and_upload, prepared_key
from typoon.runs.artifacts import ArtifactSink
from typoon.stages._workdir import workdir
from typoon.stages.prepare import RawChapterSource, prepare_chapter


async def prepare_chapter_to_archive(
    source: RawChapterSource,
    *,
    chapter_id: int,
    store: BlobStore,
    strategy: str = "auto",
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
    work: Path | None = None,
) -> tuple[str, int]:
    """Run prepare → pack `prepared.bnl` → upload. Returns (key, page_count)."""
    with workdir(work) as tmp:
        webp_dir = tmp / "prepared"
        archive_path = tmp / "prepared.bnl"

        prepare_chapter(
            source,
            webp_dir,
            strategy=strategy,
            source_label=source_label,
            artifacts=artifacts,
        )

        key = prepared_key(chapter_id)
        page_count, _locator = await pack_and_upload(
            src_dir=webp_dir,
            archive_path=archive_path,
            key=key,
            store=store,
        )
        return key, page_count
