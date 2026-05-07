"""Prepare a chapter and persist it as a Bunle archive.

`stages.prepare.prepare_chapter` writes lossless WebP pages into a temp
directory; this orchestrator packs them into `prepared.bnl` and uploads
to the chapter's prepared key. The DB pointer flip
(`Store.set_prepared_done`) is the caller's responsibility.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import pack_and_upload, prepared_key
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
    """Run prepare → pack `prepared.bnl` → upload. Returns (key, page_count)."""
    workdir_ctx = (
        _NullCtx(workdir) if workdir else tempfile.TemporaryDirectory()
    )
    with workdir_ctx as tmp_str:
        tmp = Path(tmp_str)
        webp_dir = tmp / "prepared"
        archive_path = tmp / "prepared.bnl"
        webp_dir.mkdir(parents=True, exist_ok=True)

        prepare_chapter(
            source,
            webp_dir,
            strategy=strategy,
            source_label=source_label,
            artifacts=artifacts,
        )

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
