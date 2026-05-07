"""Prepare a chapter and persist it as a Bunle archive.

Wraps `stages.prepare.prepare_chapter`. Re-encodes the produced PNGs as
WebP lossless so `bunle.pack_dir` stores them passthrough, then uploads
to the chapter's deterministic prepared key. Caller (worker loop) marks
the chapter as prepared in DB once this returns.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import pack_and_upload, prepared_key
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink
from typoon.stages.prepare import RawChapterSource, prepare_chapter


async def prepare_chapter_to_archive(
    source: RawChapterSource,
    *,
    project_id: int,
    chapter_id: int,
    store: ArtifactStore,
    strategy: str = "auto",
    source_label: str = "",
    artifacts: ArtifactSink | None = None,
    workdir: Path | None = None,
) -> tuple[str, int]:
    """Run prepare, pack the prepared archive, upload. Returns (key, page_count).

    The DB pointer flip (mark prepared done, reset render state) is the
    caller's responsibility; this function does not touch the store after
    upload.
    """
    workdir_ctx = (
        _NullCtx(workdir) if workdir else tempfile.TemporaryDirectory()
    )
    with workdir_ctx as tmp_str:
        tmp = Path(tmp_str)
        webp_dir = tmp / "prepared_webp"
        archive_path = tmp / "prepared.bnl"
        webp_dir.mkdir(parents=True, exist_ok=True)

        cp = ChapterPaths(
            projects_root=tmp / "_projects",
            slug="_prepare",
            chapter_id=chapter_id,
        )
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

        key = prepared_key(project_id, chapter_id)
        page_count = await pack_and_upload(
            src_dir=webp_dir,
            archive_path=archive_path,
            key=key,
            store=store,
        )
        return key, page_count


class _NullCtx:
    """Treat a caller-supplied workdir like a TemporaryDirectory ctx — no cleanup."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> str:
        Path(self._path).mkdir(parents=True, exist_ok=True)
        return self._path

    def __exit__(self, *_) -> None:
        pass
