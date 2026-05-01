"""Pipeline orchestrator — resume-aware, runs all stages for one chapter."""

from __future__ import annotations

import shutil
from pathlib import Path

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.session import Session
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook


async def run_chapter(
    cp: ChapterPaths,
    session: Session,
    runtime: VisionRuntime,
    *,
    redo_from: str | None = None,
    artifacts: ArtifactSink | None = None,
    hook: Hook | None = None,
) -> None:
    """Run pipeline for one chapter, skipping completed stages.

    redo_from: force re-run from this stage onward.
      Values: "prepare" | "scan" | "translate" | "render"
    """
    _redo_from(cp, redo_from)

    await _stage_prepare(cp, artifacts)
    await _stage_scan(cp, runtime, artifacts)
    await _stage_translate(cp, session, artifacts)
    _stage_render(cp, runtime, artifacts)


# ── Stage runners ─────────────────────────────────────────────────────


def _stage_prepare(cp: ChapterPaths, artifacts: ArtifactSink | None) -> None:
    if cp.is_prepared:
        return
    from typoon.adapters.local_source import LocalSource
    from typoon.stages.prepare import prepare_chapter
    prepare_chapter(LocalSource(cp.pages), cp.root,
                    source_label=str(cp.pages), artifacts=artifacts)


async def _stage_scan(
    cp: ChapterPaths,
    runtime: VisionRuntime,
    artifacts: ArtifactSink | None,
) -> None:
    if cp.is_scanned:
        return
    from typoon.domain.prepared import Chapter
    from typoon.stages.scan import scan_chapter

    prepared = Chapter.load(cp.root)
    result = scan_chapter(prepared, runtime, artifacts=artifacts)
    result.chapter.save(cp)
    result.masks.save(cp)


async def _stage_translate(
    cp: ChapterPaths,
    session: Session,
    artifacts: ArtifactSink | None,
) -> None:
    if cp.is_translated:
        return
    from typoon.domain.scan import Chapter as ScannedChapter
    from typoon.stages.translate import translate_chapter

    scanned = ScannedChapter.load(cp)
    translated = await translate_chapter(scanned, session, artifacts=artifacts)
    translated.save(cp)


def _stage_render(
    cp: ChapterPaths,
    runtime: VisionRuntime,
    artifacts: ArtifactSink | None,
) -> None:
    if cp.is_rendered:
        return
    from typoon.domain.translate import Chapter as TranslatedChapter
    from typoon.stages.render import render_chapter

    translated = TranslatedChapter.load(cp)
    masks = MaskStore.load(cp)
    render_chapter(translated, masks, runtime, out_dir=cp.root, artifacts=artifacts)


# ── Redo logic ────────────────────────────────────────────────────────

def _already_done(cp: ChapterPaths, stage: str) -> bool:
    return {
        "prepare":   cp.is_prepared,
        "scan":      cp.is_scanned,
        "translate": cp.is_translated,
        "render":    cp.is_rendered,
    }[stage]


_STAGE_ORDER = ["prepare", "scan", "translate", "render"]


def _redo_from(cp: ChapterPaths, stage: str | None) -> None:
    """Delete output files for stage and all subsequent stages."""
    if stage is None:
        return
    if stage not in _STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage!r}. Must be one of {_STAGE_ORDER}")

    idx = _STAGE_ORDER.index(stage)
    if idx <= 0:  # prepare
        if cp.manifest.exists():
            cp.manifest.unlink()
    if idx <= 1:  # scan
        if cp.scan.exists():
            cp.scan.unlink()
        if cp.masks.exists():
            shutil.rmtree(cp.masks)
    if idx <= 2:  # translate
        if cp.translate.exists():
            cp.translate.unlink()
    if idx <= 3:  # render
        if cp.render.exists():
            shutil.rmtree(cp.render)
