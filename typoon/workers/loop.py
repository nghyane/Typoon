"""Worker process entrypoint — poll DB, claim tasks, run stages.

Roles (deployment topology):
  vision     — scan + render (needs GPU + VisionRuntime)
  llm        — translate (LLM I/O, no GPU)
  api        — FastAPI server only (no worker loops)
  full       — everything in-process (dev, single-host Mac)

Each loop: claim → load → stage → save → complete.
On failure: release claim, increment attempts, log error.

Pipeline contract (RFC-001 + RFC-004):
  prepare → upload prepared.bnl + DB.set_prepared_done
  scan    → upload masks.npz + DB.save_geometry + DB.save_bubbles
  translate → DB.save_translations + DB.save_chapter_brief
  render  → upload render.bnl + DB.set_rendered(True)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from typoon.adapters.storage_registry import StorageRegistry, build_storage
from typoon.adapters.chapter_archive import masks_key, prepared_key
from typoon.adapters.ctx import TranslateCtx, make_ctx
from typoon.adapters.event_bus import EventBus, EventHook
from typoon.adapters.loader import (
    load_scanned, load_translated_with_geometry, open_prepared_reader,
)
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import Config
from typoon.paths import Paths
from typoon.runs.events import (
    CompositeHook, Hook, LoggingHook, StageDone, StageFailed, StageStarted,
)
from typoon.stages.render_archive import render_chapter_to_archive
from typoon.stages.scan import scan_chapter
from typoon.stages.translate import translate_chapter
from typoon.storage import PostgresStore, Store

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0


def _worker_id(role: str) -> str:
    """Stable worker identifier: <hostname>-<pid>-<role>-<short-uuid>.

    The hostname+pid prefix makes orphan claims diagnosable from the
    `tasks.claimed_by` column ('did host db-1 die?'). The short UUID
    suffix disambiguates multiple workers of the same role on the same
    host (e.g. several translate concurrency slots).
    """
    host = socket.gethostname() or "unknown"
    return f"{host}-{os.getpid()}-{role}-{uuid4().hex[:6]}"


class Role(StrEnum):
    vision  = "vision"
    llm     = "llm"
    api     = "api"
    storage = "storage"
    full    = "full"


# ── Loops ─────────────────────────────────────────────────────────────


async def scan_loop(
    db: Store, stores: StorageRegistry, runtime: VisionRuntime, hook: Hook,
) -> None:
    worker_id = _worker_id("scan")
    while True:
        chapter_id = await db.claim_task("scan", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        proj = await _project_for(db, chapter_id)
        await _run_scan(chapter_id, proj["id"], db, stores, runtime, hook)


async def translate_loop(
    db: Store, stores: StorageRegistry, config: Config, hook: Hook,
) -> None:
    worker_id = _worker_id("translate")
    while True:
        chapter_id = await db.claim_task("translate", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        ch   = await db.get_chapter(chapter_id)
        proj = await db.get_project(ch["project_id"])
        ctx  = make_ctx(
            project_id=proj["id"],
            chapter_id=chapter_id,
            chapter_idx=ch["idx"],
            source_lang=proj["source_lang"],
            target_lang=proj["target_lang"],
            store=db,
            config=config,
            hook=hook,
        )
        await _run_translate(chapter_id, proj["id"], ctx, db, stores, hook)


async def render_loop(
    db: Store, stores: StorageRegistry, runtime: VisionRuntime, hook: Hook,
    *, archive_salt: bytes,
) -> None:
    worker_id = _worker_id("render")
    while True:
        chapter_id = await db.claim_task("render", worker_id)
        if chapter_id is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue
        proj = await _project_for(db, chapter_id)
        await _run_render(chapter_id, proj["id"], db, stores, runtime, hook,
                          archive_salt=archive_salt)


# ── Stage runners ─────────────────────────────────────────────────────


async def _run_scan(
    chapter_id: int,
    project_id: int,
    db: Store,
    stores: StorageRegistry,
    runtime: VisionRuntime,
    hook: Hook,
) -> None:
    # Server-only artifacts (prepared.bnl, masks.npz) live on the
    # pipeline store — workers read/write them across hosts. Public
    # render archive goes to stores.public (the browser-facing backend).
    pipeline = stores.pipeline
    hook.on(StageStarted(chapter_id=chapter_id, project_id=project_id, stage="scan"))
    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
                prepared = reader.chapter()
                result = await asyncio.to_thread(
                    scan_chapter, prepared, reader, runtime,
                    chapter_id=chapter_id, project_id=project_id, hook=hook,
                )

            await db.save_geometry(chapter_id, result.geometry_records())
            await db.save_bubbles(chapter_id, result.bubble_records())

            masks_path = tmp / "masks.npz"
            result.masks.pack(masks_path)
            await pipeline.put(masks_key(project_id, chapter_id), masks_path)

        await db.advance_task(chapter_id, "scan", "translate")
        hook.on(StageDone(chapter_id=chapter_id, project_id=project_id, stage="scan"))
    except Exception as e:
        logger.exception("scan failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "scan", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, project_id=project_id, stage="scan", error=e))


async def _run_translate(
    chapter_id: int,
    project_id: int,
    ctx: TranslateCtx,
    db: Store,
    stores: StorageRegistry,
    hook: Hook,
) -> None:
    pipeline = stores.pipeline
    hook.on(StageStarted(chapter_id=chapter_id, project_id=project_id, stage="translate"))
    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
                scanned = await load_scanned(reader, db, chapter_id)
                translated, brief = await translate_chapter(scanned, reader, ctx)

        await db.save_chapter_brief(chapter_id, brief.to_dict())
        await db.save_translations(chapter_id, translated.to_db_records())
        await db.advance_task(chapter_id, "translate", "render")
        # Translation invalidates a previous render. New render task is
        # already enqueued below; the persistent flag stays True until the
        # next render finishes (so the UI keeps showing the old archive
        # while a re-render is queued).
        hook.on(StageDone(chapter_id=chapter_id, project_id=project_id, stage="translate"))
    except Exception as e:
        logger.exception("translate failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "translate", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, project_id=project_id, stage="translate", error=e))


async def _run_render(
    chapter_id: int,
    project_id: int,
    db: Store,
    stores: StorageRegistry,
    runtime: VisionRuntime,
    hook: Hook,
    *,
    archive_salt: bytes,
) -> None:
    pipeline = stores.pipeline
    public = stores.public
    hook.on(StageStarted(chapter_id=chapter_id, project_id=project_id, stage="render"))
    try:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
                translated, page_geoms = await load_translated_with_geometry(
                    reader, db, chapter_id,
                )

                masks_local = tmp / "masks.npz"
                await pipeline.get(masks_key(project_id, chapter_id), masks_local)
                masks = MaskStore.unpack(masks_local)

                locator = await render_chapter_to_archive(
                    translated,
                    project_id=project_id,
                    chapter_id=chapter_id,
                    reader=reader,
                    runtime=runtime,
                    page_geoms=page_geoms,
                    masks=masks,
                    store=public,
                    archive_salt=archive_salt,
                    work=tmp,
                    hook=hook,
                )

        await db.set_archive(chapter_id, public.backend_name, locator)
        await db.set_rendered(chapter_id, True)
        await db.complete_task(chapter_id, "render")
        hook.on(StageDone(chapter_id=chapter_id, project_id=project_id, stage="render"))
    except Exception as e:
        logger.exception("render failed chapter_id=%d", chapter_id)
        await db.fail_task(chapter_id, "render", str(e))
        hook.on(StageFailed(chapter_id=chapter_id, project_id=project_id, stage="render", error=e))


# ── Process entrypoint ────────────────────────────────────────────────


async def run_workers(
    role: Role = Role.full,
    *,
    translate_concurrency: int = 3,
    config: Config | None = None,
) -> None:
    """Start worker loops for a given role.

    All roles work against the same Postgres. `--role full` keeps
    everything in one process for dev; vision/llm/api split across hosts
    in prod via Tailscale + a shared DATABASE_URL.
    """
    from typoon.config import load_config

    config, paths = load_config() if config is None else (config, Paths())
    paths.ensure()

    db    = await PostgresStore.open(config.database_url)
    stores = build_storage(config, paths)
    archive_salt = config.storage.archive_path_salt.encode()
    loop  = asyncio.get_running_loop()
    bus   = EventBus(config.database_url)
    hook  = CompositeHook(LoggingHook(), EventHook(bus, loop))

    runtime: VisionRuntime | None = None
    if role in (Role.vision, Role.full):
        runtime = VisionRuntime.from_config(config)[0]

    loops: list = []
    if role in (Role.vision, Role.full):
        assert runtime is not None
        loops.append(scan_loop(db, stores, runtime, hook))
        loops.append(render_loop(db, stores, runtime, hook, archive_salt=archive_salt))
    if role in (Role.llm, Role.full):
        loops.extend(
            translate_loop(db, stores, config, hook)
            for _ in range(translate_concurrency)
        )

    if not loops:
        logger.info("role=%s: no worker loops to run", role)
        await stores.aclose()
        await bus.close()
        await db.close()
        return

    try:
        await asyncio.gather(*loops)
    finally:
        await stores.aclose()
        await bus.close()
        await db.close()


# ── Helpers ───────────────────────────────────────────────────────────


async def _project_for(db: Store, chapter_id: int) -> dict:
    ch = await db.get_chapter(chapter_id)
    return await db.get_project(ch["project_id"])
