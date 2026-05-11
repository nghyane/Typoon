"""Worker process — claims pipeline tasks and runs them.

Worker model: a single asyncio process owns one connection-multiplexed
pool of stage runners (prepare/scan/translate/render). Each stage waits
on Postgres LISTEN for its channel; when a task lands, the worker
claims it via `claim_task` and runs the corresponding handler.

Wakeup latency is ~5 ms (LISTEN/NOTIFY); a 30-second safety poll covers
the case where a notification is missed (network blip, dropped
connection). No busy polling, near-zero idle CPU.

Restart resilience: every worker process picks a stable
`{hostname}-{pid}` prefix on boot and immediately calls
`release_claims_by_prefix` to clear any claim left behind by a previous
PID on the same host (graceful exit + SIGKILL alike). Cross-host crashes
are still handled by the existing `STALE_CLAIM_SECONDS` reaper inside
`claim_task` — that's the only generic way to detect a dead remote
process.

Pipeline contract (RFC-001 + RFC-004):
  prepare   → fetch zip from inbox → unpack → upload prepared.bnl
              + DB.set_prepared_done
  scan      → upload masks.npz + DB.save_geometry + DB.save_bubbles
  translate → DB.save_translations + DB.save_chapter_brief
  render    → upload render.bnl + DB.set_rendered(True)

Roles (deployment topology):
  vision     — prepare + scan + render (needs ANE/GPU + VisionRuntime)
  llm        — prepare + translate (LLM I/O, no GPU)
  api        — FastAPI server only (no worker loops)
  full       — everything in-process (dev, single-host Mac)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import asyncpg

from typoon.adapters.channel_bus import ChannelBus, ChannelHook
from typoon.adapters.chapter_archive import masks_key, prepared_key
from typoon.adapters.ctx import make_ctx
from typoon.adapters.loader import (
    load_scanned, load_translated_with_geometry, open_prepared_reader,
)
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.storage_registry import StorageRegistry, build_storage
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import Config
from typoon.llm import TransientCredentialError, UpstreamUnavailable
from typoon.paths import Paths
from typoon.runs.events import (
    CompositeHook, Event, Hook, LoggingHook, PageDone, StageDone,
    StageFailed, StageStarted,
)
from typoon.stages.render_archive import render_chapter_to_archive
from typoon.stages.scan import scan_chapter
from typoon.stages.translate import translate_chapter
from typoon.storage import PostgresStore, Store

logger = logging.getLogger(__name__)


# ── Identity ────────────────────────────────────────────────────────


def _host_prefix() -> str:
    """Stable per-process prefix for `tasks.claimed_by`.

    `{hostname}-{pid}` is enough to detect ghost claims left by an
    earlier PID on this host: when a fresh process boots it can call
    `release_claims_by_prefix(_host_prefix())` and clear whatever its
    predecessor left dangling. Cross-host crashes still fall through
    to the staleness reaper inside `claim_task`.
    """
    host = socket.gethostname() or "unknown"
    return f"{host}-{os.getpid()}"


def _worker_id(stage: str) -> str:
    """`{hostname}-{pid}-{stage}-{shortuuid}` — diagnosable in logs."""
    return f"{_host_prefix()}-{stage}-{uuid4().hex[:6]}"


# ── Hooks ──────────────────────────────────────────────────────────


class ProgressPersistingHook(Hook):
    """Persist PageDone progress onto chapters row for UI replay.

    The channel bus only delivers live events; clients that connect
    mid-render need somewhere durable to read the current
    (stage, index, total) tuple from. Stages may emit from worker
    threads; we schedule the DB call on the event loop captured at
    construction.
    """

    def __init__(self, db: Store, loop: asyncio.AbstractEventLoop) -> None:
        self._db = db
        self._loop = loop

    def on(self, event: Event) -> None:
        if not isinstance(event, PageDone) or event.chapter_id <= 0:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._db.set_chapter_progress(
                    event.chapter_id,
                    stage=event.stage,
                    index=event.page_index,
                    total=event.page_total,
                ),
                self._loop,
            )
        except RuntimeError:
            # Loop closed during shutdown — nothing to persist.
            pass


# ── Roles ──────────────────────────────────────────────────────────


class Role(StrEnum):
    vision  = "vision"
    llm     = "llm"
    api     = "api"
    storage = "storage"
    full    = "full"


# Sleep this long before letting the pump re-claim after a transient
# upstream failure. Long enough to ride out a brief credential rotation
# and avoid hot-looping the broken endpoint; short enough that a fixed
# config reaches the next claim within a chapter's editing window.
_REQUEUE_BACKOFF_SECONDS = 60.0


_STAGES_BY_ROLE: dict[Role, tuple[str, ...]] = {
    Role.vision:  ("prepare", "scan", "render"),
    Role.llm:     ("prepare", "translate"),
    Role.full:    ("prepare", "scan", "translate", "render"),
    Role.api:     (),
    Role.storage: (),
}


# ── Stage handler signature ─────────────────────────────────────────


@dataclass
class StageContext:
    """Everything a stage handler needs, bundled once per process.

    Built in `run_workers` and shared across every claim. Handlers are
    pure functions of `(ctx, chapter_id)` — no globals, no module-level
    state. Stage-specific extras (VisionRuntime, inbox, archive_salt)
    live on the context so the dispatch loop is generic.
    """
    db:           Store
    stores:       StorageRegistry
    hook:         Hook
    paths:        Paths
    config:       Config
    archive_salt: bytes
    runtime:      VisionRuntime | None = None
    inbox:        object | None = None


StageHandler = Callable[[StageContext, int, int], Awaitable[None]]


# ── Stage handlers ──────────────────────────────────────────────────


async def _handle_prepare(ctx: StageContext, chapter_id: int, project_id: int) -> None:
    from typoon.sources.upload import UnpackError, unpack_zip
    from typoon.stages.prepare_archive import prepare_chapter_to_archive
    from typoon.sources.local import LocalSource

    handle = await ctx.db.get_inbox_handle(chapter_id)
    if handle is None:
        raise RuntimeError(
            f"prepare claim {chapter_id}: missing inbox handle "
            f"(was the chapter created without queue_chapter?)",
        )

    with tempfile.TemporaryDirectory(prefix="typoon-prepare-") as tmp_str:
        tmp = Path(tmp_str)
        pages_dir = tmp / "pages"

        # complete_multipart is idempotent: if a prior attempt already
        # completed it, S3 returns NoSuchUpload and we proceed to fetch.
        try:
            await ctx.inbox.complete_multipart(  # type: ignore[union-attr]
                tmp_id=handle.tmp_id,
                upload_id=handle.upload_id,
                parts=list(handle.parts),
            )
        except Exception as e:
            if "NoSuchUpload" not in str(e):
                raise

        zip_path = tmp / "chapter.zip"
        size = await ctx.inbox.fetch(tmp_id=handle.tmp_id, dest=zip_path)  # type: ignore[union-attr]
        if size <= 0:
            raise RuntimeError(f"prepare {chapter_id}: empty zip from inbox")
        try:
            n_unpacked = unpack_zip(zip_path.read_bytes(), pages_dir)
        except UnpackError as e:
            raise RuntimeError(f"prepare {chapter_id}: unpack failed: {e}") from e
        if n_unpacked == 0:
            raise RuntimeError(f"prepare {chapter_id}: zip contained no pages")

        _key, page_count = await prepare_chapter_to_archive(
            LocalSource(pages_dir),
            project_id=project_id, chapter_id=chapter_id,
            store=ctx.stores.pipeline,
            strategy="auto",
            work=tmp,
        )

    await ctx.db.set_prepared_done(chapter_id, page_count)
    await ctx.db.advance_task(chapter_id, "prepare", "scan")
    await ctx.db.clear_inbox_handle(chapter_id)

    # The bucket lifecycle rule sweeps anything we miss.
    try:
        await ctx.inbox.delete(tmp_id=handle.tmp_id)  # type: ignore[union-attr]
    except Exception as e:
        logger.warning(
            "inbox cleanup failed (chapter=%d tmp=%s): %s",
            chapter_id, handle.tmp_id, e,
        )


async def _handle_scan(ctx: StageContext, chapter_id: int, project_id: int) -> None:
    assert ctx.runtime is not None, "scan requires VisionRuntime"
    proj = await ctx.db.get_project(project_id)
    source_lang = proj["source_lang"]

    pipeline = ctx.stores.pipeline
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
            prepared = reader.chapter()
            result = await asyncio.to_thread(
                scan_chapter, prepared, reader, ctx.runtime,
                source_lang=source_lang,
                chapter_id=chapter_id, project_id=project_id, hook=ctx.hook,
            )

        await ctx.db.save_geometry(chapter_id, result.geometry_records())
        await ctx.db.save_bubbles(chapter_id, result.bubble_records())

        masks_path = tmp / "masks.npz"
        result.masks.pack(masks_path)
        await pipeline.put(masks_key(project_id, chapter_id), masks_path)

    await ctx.db.advance_task(chapter_id, "scan", "translate")


async def _handle_translate(ctx: StageContext, chapter_id: int, project_id: int) -> None:
    ch    = await ctx.db.get_chapter(chapter_id)
    proj  = await ctx.db.get_project(project_id)
    tctx  = make_ctx(
        project_id=project_id,
        chapter_id=chapter_id,
        chapter_position=ch["position"],
        source_lang=proj["source_lang"],
        target_lang=proj["target_lang"],
        store=ctx.db,
        config=ctx.config,
        hook=ctx.hook,
    )

    pipeline = ctx.stores.pipeline
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
            scanned = await load_scanned(reader, ctx.db, chapter_id)
            translated, brief = await translate_chapter(scanned, reader, tctx)

    await ctx.db.save_chapter_brief(chapter_id, brief.to_dict())
    await ctx.db.save_translations(chapter_id, translated.to_db_records())
    await ctx.db.advance_task(chapter_id, "translate", "render")


async def _handle_render(ctx: StageContext, chapter_id: int, project_id: int) -> None:
    assert ctx.runtime is not None, "render requires VisionRuntime"
    pipeline = ctx.stores.pipeline
    public   = ctx.stores.public

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(pipeline, prepared_key(project_id, chapter_id), tmp) as reader:
            translated, page_geoms = await load_translated_with_geometry(
                reader, ctx.db, chapter_id,
            )

            masks_local = tmp / "masks.npz"
            await pipeline.get(masks_key(project_id, chapter_id), masks_local)
            masks = MaskStore.unpack(masks_local)

            # Pages flagged as full-page noise by the context agent are
            # dropped from the public render archive. Brief is always
            # present at render time (translate is a hard prerequisite).
            brief_dict = await ctx.db.get_chapter_brief(chapter_id)
            skip_pages: frozenset[int] = frozenset(
                int(p) for p in (brief_dict or {}).get("noise_pages", [])
            )

            locator, public_page_count = await render_chapter_to_archive(
                translated,
                project_id=project_id,
                chapter_id=chapter_id,
                reader=reader,
                runtime=ctx.runtime,
                page_geoms=page_geoms,
                masks=masks,
                store=public,
                archive_salt=ctx.archive_salt,
                work=tmp,
                hook=ctx.hook,
                skip_pages=skip_pages,
            )

    await ctx.db.set_archive(chapter_id, public.backend_name, locator)
    await ctx.db.set_rendered(chapter_id, True, page_count=public_page_count)
    await ctx.db.complete_task(chapter_id, "render")


_HANDLERS: dict[str, StageHandler] = {
    "prepare":   _handle_prepare,
    "scan":      _handle_scan,
    "translate": _handle_translate,
    "render":    _handle_render,
}


# ── Stage pump ─────────────────────────────────────────────────────


async def _run_one(
    ctx:        StageContext,
    stage:      str,
    chapter_id: int,
) -> None:
    """Dispatch one claim through its handler with uniform lifecycle.

    `StageStarted` / `StageDone` / `StageFailed` are emitted here, not
    in the handlers, so adding a new stage means writing one handler
    function and registering it in `_HANDLERS`.
    """
    ch = await ctx.db.get_chapter(chapter_id)
    if ch is None:
        # Chapter vanished between claim and dispatch (rare race: user
        # delete + worker pick up). The stale `tasks` row disappears
        # with the chapter via ON DELETE CASCADE — nothing to do.
        return
    project_id = ch["project_id"]

    ctx.hook.on(StageStarted(chapter_id=chapter_id, project_id=project_id, stage=stage))
    try:
        await _HANDLERS[stage](ctx, chapter_id, project_id)
        ctx.hook.on(StageDone(chapter_id=chapter_id, project_id=project_id, stage=stage))
    except (TransientCredentialError, UpstreamUnavailable) as e:
        # Provider is sick (token revoked, credential pool empty,
        # gateway 5xx). Don't burn an attempt — release the claim and
        # let the next pump cycle pick the chapter up. Sleep first so
        # we don't immediately re-claim and hammer the same broken
        # endpoint while the operator is still rotating credentials.
        logger.warning(
            "%s requeue chapter_id=%d (%s): %s",
            stage, chapter_id, type(e).__name__, e,
        )
        await ctx.db.requeue_task(chapter_id, stage, str(e))
        ctx.hook.on(StageFailed(
            chapter_id=chapter_id, project_id=project_id, stage=stage, error=e,
        ))
        await asyncio.sleep(_REQUEUE_BACKOFF_SECONDS)
    except Exception as e:
        logger.exception("%s failed chapter_id=%d", stage, chapter_id)
        await ctx.db.fail_task(chapter_id, stage, str(e))
        ctx.hook.on(StageFailed(
            chapter_id=chapter_id, project_id=project_id, stage=stage, error=e,
        ))


async def _stage_pump(
    ctx:    StageContext,
    stage:  str,
    dsn:    str,
    stop:   asyncio.Event,
) -> None:
    """Run claims for one stage until cancelled.

    Wake-up sources, in order of preference:

      1. Postgres NOTIFY on `typoon_task_<stage>` — fires within ~5 ms
         of any insert/release on the `tasks` table.
      2. 30-second safety poll — covers missed notifications (the LISTEN
         connection dropping while we were away, etc.). Long enough to
         keep CPU near zero; short enough that a stuck worker recovers
         without operator intervention.

    The pump drains the queue greedily on every wake-up: claim until
    `claim_task` returns None, then sleep again. This keeps the
    notification firehose to one signal regardless of how many tasks
    landed in a burst.
    """
    worker_id = _worker_id(stage)
    channel = f"typoon_task_{stage}"
    notif = asyncio.Event()

    def _on_notify(_conn, _pid, _chan, _payload):
        notif.set()

    listen_conn = await asyncpg.connect(dsn)
    try:
        await listen_conn.add_listener(channel, _on_notify)
        logger.info("[%s] pump up on LISTEN %s", stage, channel)

        # Drain immediately on boot — claims released by startup reaper
        # are waiting for us.
        while not stop.is_set():
            claimed_any = True
            while claimed_any and not stop.is_set():
                claimed_any = False
                while not stop.is_set():
                    chapter_id = await ctx.db.claim_task(stage, worker_id)
                    if chapter_id is None:
                        break
                    claimed_any = True
                    await _run_one(ctx, stage, chapter_id)
            if stop.is_set():
                break

            # Wait for either a NOTIFY or the safety-net poll, whichever
            # comes first. `asyncio.wait` returns when any task in the
            # set finishes, so a single NOTIFY wakes us instantly even
            # if shutdown isn't requested. Outstanding tasks get
            # cancelled to avoid leaking handles.
            notif.clear()
            wakeup = asyncio.create_task(notif.wait())
            quit_  = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait(
                {wakeup, quit_},
                timeout=30,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
    finally:
        try:
            await listen_conn.remove_listener(channel, _on_notify)
        except Exception:
            pass
        await listen_conn.close()
        logger.info("[%s] pump down", stage)


# ── Process entrypoint ────────────────────────────────────────────


async def run_workers(
    role: Role = Role.full,
    *,
    translate_concurrency: int = 3,
    config: Config | None = None,
) -> None:
    """Start every stage pump for `role` and block until shutdown.

    Lifecycle:

      1. Bind SIGTERM/SIGINT to a `stop_event` so launchd's stop signal
         (or a Ctrl-C in dev) propagates as cooperative cancellation.
      2. Release any claim left by a previous PID on this host — a hard
         exit (SIGKILL, kernel panic, machine reboot) bypasses the
         stop_event path, so the freshly-booted worker has to clear the
         dust on startup. Cross-host crashes are still handled by the
         staleness reaper inside `claim_task`.
      3. Build a single `StageContext` shared by every pump.
      4. Spawn one `_stage_pump` per active stage inside an asyncio
         TaskGroup. Translate is the only stage with parallel slots
         (LLM I/O, no GPU/ANE) — extra pumps share the same handler.
      5. On `stop_event`, every pump unwinds; pending claims drain
         naturally before the TaskGroup exits.
    """
    from typoon.config import load_config
    from typoon.adapters.inbox import build_inbox

    config, paths = load_config() if config is None else (config, Paths())
    paths.ensure()

    db    = await PostgresStore.open(config.database_url)
    stores = build_storage(config, paths)
    archive_salt = config.storage.archive_path_salt.encode()
    loop  = asyncio.get_running_loop()
    bus   = ChannelBus(config.database_url)
    hook  = CompositeHook(
        LoggingHook(),
        ChannelHook(bus, loop),
        ProgressPersistingHook(db, loop),
    )

    runtime: VisionRuntime | None = None
    if role in (Role.vision, Role.full):
        runtime = VisionRuntime.from_config(config)[0]

    inbox = build_inbox(
        config.storage,
        paths_root=paths.artifacts,
        base_url=config.server.public_base_url,
    )

    ctx = StageContext(
        db=db, stores=stores, hook=hook, paths=paths, config=config,
        archive_salt=archive_salt, runtime=runtime, inbox=inbox,
    )

    stages = list(_STAGES_BY_ROLE.get(role, ()))
    if not stages:
        logger.info("role=%s: no worker loops to run", role)
        await stores.aclose()
        await bus.close()
        await db.close()
        return

    # Startup: clear ghost claims left by a previous PID on this host.
    released = await db.release_claims_by_prefix(_host_prefix())
    if released:
        logger.info("released %d ghost claim(s) from prior PID(s) on this host", released)

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows: signals on the loop aren't supported.
            pass

    try:
        async with asyncio.TaskGroup() as tg:
            for stage in stages:
                pumps = translate_concurrency if stage == "translate" else 1
                for _ in range(pumps):
                    tg.create_task(
                        _stage_pump(ctx, stage, config.database_url, stop),
                    )
            # Hold the TaskGroup open until shutdown is requested. The
            # pumps themselves exit cleanly on stop.set(); we wait here
            # so the TaskGroup doesn't unwind via a stray pump returning.
            await stop.wait()
    finally:
        await stores.aclose()
        await bus.close()
        await db.close()
