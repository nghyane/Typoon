"""Worker process entrypoint — poll DB, claim tasks, run stages.

Three independent loops, each owning one stage:
  scan      — exclusive VisionRuntime.scanner (1 worker)
  translate — concurrent LLM I/O (N workers)
  render    — exclusive VisionRuntime.eraser  (1 worker)

Each loop: claim → load → stage → save → complete.
On failure: release claim, increment attempts, log error.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from typoon.adapters.ctx import TranslateCtx, make_ctx
from typoon.adapters.loader import load_prepared, load_scanned, load_translated_with_geometry
from typoon.adapters.mask_store import save_scan_geometry
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.paths import Paths, ProjectPaths, ChapterPaths
from typoon.runs.events import Hook, StageDone, StageFailed, StageStarted
from typoon.stages.scan import scan_chapter
from typoon.stages.translate import translate_chapter
from typoon.stages.render import render_chapter
from typoon.storage.sqlite import SqliteStore

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0


async def scan_loop(db: SqliteStore, runtime: VisionRuntime, paths: Paths, hook: Hook) -> None:
    worker_id = str(uuid4())
    while True:
        chapter_id = await db.claim_task("scan", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        cp, _ch, _proj = await _resolve(chapter_id, db, paths)
        await _run_scan(cp, chapter_id, db, runtime, hook)


async def translate_loop(db: SqliteStore, paths: Paths, config, hook: Hook) -> None:
    worker_id = str(uuid4())
    while True:
        chapter_id = await db.claim_task("translate", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        cp, ch, proj = await _resolve(chapter_id, db, paths)
        ctx = make_ctx(
            project_id=proj["id"],
            chapter_id=chapter_id,
            chapter_idx=ch["idx"],
            source_lang=proj["source_lang"],
            target_lang=proj["target_lang"],
            store=db,
            config=config,
            hook=hook,
        )
        await _run_translate(cp, chapter_id, ctx, db, hook)


async def render_loop(db: SqliteStore, runtime: VisionRuntime, paths: Paths, hook: Hook) -> None:
    worker_id = str(uuid4())
    while True:
        chapter_id = await db.claim_task("render", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        cp, _ch, _proj = await _resolve(chapter_id, db, paths)
        await _run_render(cp, chapter_id, db, runtime, hook)


# ── Stage runners ─────────────────────────────────────────────────────


async def _run_scan(
    cp: ChapterPaths,
    chapter_id: int,
    db: SqliteStore,
    runtime: VisionRuntime,
    hook: Hook,
) -> None:
    hook.on(StageStarted(chapter_id=chapter_id, stage="scan"))
    try:
        prepared = load_prepared(cp)
        result   = await asyncio.to_thread(scan_chapter, prepared, runtime)
        save_scan_geometry(cp, result.geometry)
        result.masks.save(cp)
        await db.save_bubbles(chapter_id, result.bubble_records())
        await db.complete_task(chapter_id, "scan")
        await db.enqueue(chapter_id, "translate")
        hook.on(StageDone(chapter_id=chapter_id, stage="scan"))
    except Exception as e:
        logger.exception("scan failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "scan", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, stage="scan", error=e))


async def _run_translate(
    cp: ChapterPaths,
    chapter_id: int,
    ctx: TranslateCtx,
    db: SqliteStore,
    hook: Hook,
) -> None:
    hook.on(StageStarted(chapter_id=chapter_id, stage="translate"))
    try:
        scanned             = await load_scanned(cp, db, chapter_id)
        translated, brief   = await translate_chapter(scanned, ctx)
        await db.save_chapter_brief(chapter_id, brief.to_dict())
        await db.save_translations(chapter_id, translated.to_db_records())
        await db.complete_task(chapter_id, "translate")
        await db.enqueue(chapter_id, "render")
        hook.on(StageDone(chapter_id=chapter_id, stage="translate"))
    except Exception as e:
        logger.exception("translate failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "translate", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, stage="translate", error=e))


async def _run_render(
    cp: ChapterPaths,
    chapter_id: int,
    db: SqliteStore,
    runtime: VisionRuntime,
    hook: Hook,
) -> None:
    hook.on(StageStarted(chapter_id=chapter_id, stage="render"))
    try:
        translated, page_geoms = await load_translated_with_geometry(cp, db, chapter_id)
        await asyncio.to_thread(render_chapter, translated, cp, runtime, page_geoms)
        await db.complete_task(chapter_id, "render")
        hook.on(StageDone(chapter_id=chapter_id, stage="render"))
    except Exception as e:
        logger.exception("render failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "render", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, stage="render", error=e))


# ── Process entrypoint ────────────────────────────────────────────────


async def run_workers(*, translate_concurrency: int = 3, config=None) -> None:
    from typoon.config import load_config

    config, paths = load_config() if config is None else (config, Paths())
    paths.ensure()

    db      = await SqliteStore.open(paths.db)
    runtime = VisionRuntime.from_config(config)[0]
    hook    = Hook()

    try:
        await asyncio.gather(
            scan_loop(db, runtime, paths, hook),
            render_loop(db, runtime, paths, hook),
            *[translate_loop(db, paths, config, hook) for _ in range(translate_concurrency)],
        )
    finally:
        await db.close()


# ── Helper ────────────────────────────────────────────────────────────


async def _resolve(
    chapter_id: int,
    db: SqliteStore,
    paths: Paths,
) -> tuple[ChapterPaths, dict, dict]:
    """Single DB round-trip: return (ChapterPaths, chapter_row, project_row)."""
    ch   = await db.get_chapter(chapter_id)
    proj = await db.get_project(ch["project_id"])
    cp   = ProjectPaths(paths.projects, proj["slug"]).chapter(chapter_id)
    return cp, ch, proj
