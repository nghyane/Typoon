"""Pipeline stage execution with resume support."""

from __future__ import annotations

import shutil
from enum import Enum

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook, StageDone, StageStarted, StageFailed


class Stage(str, Enum):
    PREPARE   = "prepare"
    SCAN      = "scan"
    TRANSLATE = "translate"
    RENDER    = "render"

    def is_done(self, cp: ChapterPaths) -> bool:
        return {
            Stage.PREPARE:   cp.is_prepared,
            Stage.SCAN:      cp.is_scanned,
            Stage.TRANSLATE: cp.is_translated,
            Stage.RENDER:    cp.is_rendered,
        }[self]

    def outputs(self, cp: ChapterPaths) -> list:
        return {
            Stage.PREPARE:   [cp.manifest],
            Stage.SCAN:      [cp.scan, cp.masks],
            Stage.TRANSLATE: [cp.translate],
            Stage.RENDER:    [cp.render],
        }[self]


def redo_from(cp: ChapterPaths, stage: Stage | str | None) -> None:
    """Delete output files for stage and all subsequent stages."""
    if stage is None:
        return
    if isinstance(stage, str):
        stage = Stage(stage)
    stages = list(Stage)
    for s in stages[stages.index(stage):]:
        for path in s.outputs(cp):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink(missing_ok=True)


async def run(
    cp: ChapterPaths,
    ctx: TranslateCtx,
    runtime: VisionRuntime,
    *,
    hook: Hook,
    redo: Stage | str | None = None,
    artifacts: ArtifactSink | None = None,
) -> None:
    """Run all pipeline stages for one chapter. Raises on failure."""
    redo_from(cp, redo)
    await _run_prepare(cp, hook, artifacts)
    await _run_scan(cp, runtime, hook, artifacts)
    await _run_translate(cp, ctx, hook, artifacts)
    await _run_render(cp, runtime, hook, artifacts)


# ── Stage runners ─────────────────────────────────────────────────────


async def _run_prepare(
    cp: ChapterPaths, hook: Hook, artifacts: ArtifactSink | None,
) -> None:
    if Stage.PREPARE.is_done(cp):
        return
    hook.on(StageStarted(idx=cp.idx, stage=Stage.PREPARE.value))
    try:
        from typoon.sources.local import LocalSource
        from typoon.stages.prepare import prepare_chapter
        prepare_chapter(LocalSource(cp.pages), cp.root,
                        source_label=str(cp.pages), artifacts=artifacts)
    except Exception as e:
        hook.on(StageFailed(idx=cp.idx, stage=Stage.PREPARE.value, error=e))
        raise
    hook.on(StageDone(idx=cp.idx, stage=Stage.PREPARE.value))


async def _run_scan(
    cp: ChapterPaths, runtime: VisionRuntime, hook: Hook, artifacts: ArtifactSink | None,
) -> None:
    if Stage.SCAN.is_done(cp):
        return
    hook.on(StageStarted(idx=cp.idx, stage=Stage.SCAN.value))
    try:
        from typoon.domain.prepared import Chapter
        from typoon.stages.scan import scan_chapter
        result = scan_chapter(Chapter.load(cp.root), runtime, artifacts=artifacts)
        result.chapter.save(cp)
        result.masks.save(cp)
    except Exception as e:
        hook.on(StageFailed(idx=cp.idx, stage=Stage.SCAN.value, error=e))
        raise
    hook.on(StageDone(idx=cp.idx, stage=Stage.SCAN.value))


async def _run_translate(
    cp: ChapterPaths, ctx: TranslateCtx, hook: Hook, artifacts: ArtifactSink | None,
) -> None:
    if Stage.TRANSLATE.is_done(cp):
        return
    hook.on(StageStarted(idx=cp.idx, stage=Stage.TRANSLATE.value))
    try:
        from typoon.domain.scan import Chapter as ScannedChapter
        from typoon.stages.translate import translate_chapter
        translated = await translate_chapter(ScannedChapter.load(cp), ctx, artifacts=artifacts)
        translated.save(cp)
    except Exception as e:
        hook.on(StageFailed(idx=cp.idx, stage=Stage.TRANSLATE.value, error=e))
        raise
    hook.on(StageDone(idx=cp.idx, stage=Stage.TRANSLATE.value))


async def _run_render(
    cp: ChapterPaths, runtime: VisionRuntime, hook: Hook, artifacts: ArtifactSink | None,
) -> None:
    if Stage.RENDER.is_done(cp):
        return
    hook.on(StageStarted(idx=cp.idx, stage=Stage.RENDER.value))
    try:
        from typoon.domain.translate import Chapter as TranslatedChapter
        from typoon.stages.render import render_chapter
        render_chapter(TranslatedChapter.load(cp), MaskStore.load(cp), runtime, render_dir=cp.render)
    except Exception as e:
        hook.on(StageFailed(idx=cp.idx, stage=Stage.RENDER.value, error=e))
        raise
    hook.on(StageDone(idx=cp.idx, stage=Stage.RENDER.value))
