"""Prepare a chapter and persist it as a Bunle archive.

Wraps `stages.prepare.prepare_chapter` without modifying it. The existing
function writes PNGs into a directory; this orchestrator re-encodes those
to WebP lossless (so `bunle.pack_dir` can passthrough without re-encoding)
and uploads the resulting `prepared-<rev>.bnl`.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import (
    ArchiveResult,
    pack_and_upload,
    prepared_key,
)
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink
from typoon.stages.prepare import RawChapterSource, prepare_chapter


@dataclass(frozen=True)
class PreparedArtifact:
    archive: ArchiveResult
    previous_key: str | None


async def prepare_chapter_to_archive(
    source: RawChapterSource,
    *,
    project_id: int,
    chapter_id: int,
    store: ArtifactStore,
    set_prepared_artifact,
    strategy: str = "auto",
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
    workdir: Path | None = None,
) -> PreparedArtifact:
    """Run prepare into a tmp dir, pack a Bunle archive, upload, flip DB pointer.

    `set_prepared_artifact` is `Store.set_prepared_artifact`. Pulled in via
    parameter so this stage stays decoupled from the storage import path.
    """
    workdir_ctx = (
        _NullCtx(workdir) if workdir else tempfile.TemporaryDirectory()
    )
    with workdir_ctx as tmp_str:
        tmp = Path(tmp_str)
        png_dir = tmp / "prepared_png"
        webp_dir = tmp / "prepared_webp"
        archive_path = tmp / "prepared.bnl"
        png_dir.mkdir(parents=True, exist_ok=True)
        webp_dir.mkdir(parents=True, exist_ok=True)

        cp = ChapterPaths(
            projects_root=tmp / "_projects",
            slug="_prepare",
            chapter_id=chapter_id,
        )
        # prepare_chapter writes to cp.pages — point that at our png_dir
        cp.pages.mkdir(parents=True, exist_ok=True)
        chapter = prepare_chapter(
            source,
            cp,
            strategy=strategy,
            source_label=source_label,
            artifacts=artifacts,
        )

        for page in chapter.pages:
            src_png = cp.page(page.index)
            dst_webp = webp_dir / f"{page.index:04d}.webp"
            with Image.open(src_png) as img:
                img.save(dst_webp, format="WEBP", lossless=True, quality=100)

        result = await pack_and_upload(
            src_dir=webp_dir,
            archive_path=archive_path,
            key_builder=lambda rev: prepared_key(project_id, chapter_id, rev),
            store=store,
        )

        previous = await set_prepared_artifact(
            chapter_id,
            prepared_key=result.key,
            page_count=result.page_count,
        )
        if previous and previous != result.key:
            await store.delete(previous)

        return PreparedArtifact(archive=result, previous_key=previous)


class _NullCtx:
    """Treat a caller-supplied workdir like a TemporaryDirectory ctx — no cleanup."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> str:
        Path(self._path).mkdir(parents=True, exist_ok=True)
        return self._path

    def __exit__(self, *_) -> None:
        pass
