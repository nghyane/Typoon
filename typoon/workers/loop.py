"""Worker process — claims pipeline tasks and runs them.

3-stage cache flow (per RFC v5):

  Layer 1 (chapter scope)
    prepare    chapter_id → pack pages from inbox into prepared.bnl,
                            set chapter.prepared_hash/locator. Skipped
                            when prepared_hash already set (CAS hit).
    scan       chapter_id → OCR + geometry + masks bound to chapter.
                            Shared across every translation that
                            references this chapter.

  Layer 2 (draft scope, cross-user via visibility)
    translate  draft_id → LLM run filling translation_draft_bubbles
                          and draft_briefs. Other users with matching
                          (chapter, src, tgt, glossary_fp) reuse this
                          draft via find_reusable_draft → cache hit.

  Layer 3 (translation scope, per-user fallback)
    render     draft_id        → default render.bnl shared by every
                                  translation pointing at the draft
                                  with no edits.
                 OR
               translation_id  → per-user render.bnl when sparse
                                 edits diverge from the draft.

Wakeup model: one asyncio process per role; each stage has its own
`_stage_pump` listening on `pg_notify('typoon_task_<stage>')`. LISTEN
payload format is `'<target_kind>:<target_id>'`; pumps fan out work
to per-stage handlers based on the claimed task tuple.

Restart resilience: every process picks a stable `{hostname}-{pid}`
prefix on boot and calls `release_claims_by_prefix` to clear claims
left by a previous PID on this host. Cross-host crashes are handled
by `STALE_CLAIM_SECONDS` inside `claim_task`.
"""

from __future__ import annotations

import asyncio
import hashlib
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
from typoon.adapters.inbox import ChapterInbox, build_inbox
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
    """Stable per-process prefix for `tasks.claimed_by`."""
    host = socket.gethostname() or "unknown"
    return f"{host}-{os.getpid()}"


def _worker_id(stage: str) -> str:
    """`{hostname}-{pid}-{stage}-{shortuuid}` — diagnosable in logs."""
    return f"{_host_prefix()}-{stage}-{uuid4().hex[:6]}"


# ── Hooks ──────────────────────────────────────────────────────────


class ProgressPersistingHook(Hook):
    """Persist PageDone progress onto translation_drafts row for UI replay.

    Scan/translate/render are all draft-keyed for progress (scan runs
    chapter-wide but progress UI naturally surfaces under whichever
    draft triggered the spawn). Chapter-level scan progress is shared
    across every draft awaiting the chapter — we update each pending
    draft on this chapter so any UI subscribed to a draft sees it.
    """

    def __init__(self, db: Store, loop: asyncio.AbstractEventLoop) -> None:
        self._db = db
        self._loop = loop

    def on(self, event: Event) -> None:
        if not isinstance(event, PageDone):
            return
        draft_id = getattr(event, "draft_id", 0)
        if not draft_id:
            return  # Chapter-level events don't persist progress; UI
                    # polls the draft row directly.
        try:
            asyncio.run_coroutine_threadsafe(
                self._db.set_draft_progress(
                    draft_id,
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


# ── Stage context (shared by handlers) ─────────────────────────────


@dataclass
class StageContext:
    db:           Store
    stores:       StorageRegistry
    hook:         Hook
    paths:        Paths
    config:       Config
    archive_salt: bytes
    runtime:      VisionRuntime | None = None
    inbox:        ChapterInbox | None = None


# Stage handler signature: receives the context + (target_kind, target_id).
# Handlers are pure functions; events / DB updates flow through ctx.
StageHandler = Callable[[StageContext, str, int], Awaitable[None]]


# ── Helpers shared by handlers ─────────────────────────────────────


async def _chapter_id_for_target(
    db: Store, target_kind: str, target_id: int,
) -> int | None:
    """Resolve the chapter that a (kind, id) ultimately belongs to.

    prepare/scan tasks key on chapter directly; translate keys on
    draft → draft.chapter_id; render targets a translation but its
    pixels still live on the draft's chapter (translation rows sit
    at Work-chapter scope, not per-pixel).
    """
    if target_kind == "chapter":
        return target_id
    if target_kind == "draft":
        d = await db.get_draft(target_id)
        return d["chapter_id"] if d else None
    if target_kind == "translation":
        t = await db.get_translation(target_id)
        if t is None or t.get("draft_id") is None:
            return None
        d = await db.get_draft(t["draft_id"])
        return d["chapter_id"] if d else None
    return None


def _hash_file(path: Path) -> str:
    """SHA256 (hex) of a prepared.bnl — the CAS key."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Stage handlers ──────────────────────────────────────────────────


async def _handle_prepare(
    ctx: StageContext, target_kind: str, target_id: int,
) -> None:
    """Prepare stage runs once per chapter and writes prepared.bnl.

    CAS dedup: if a chapter with the same prepared_hash already has a
    locator, reuse it (skip the pack + upload) and just set the
    pointer on this chapter. Common after the same file gets uploaded
    twice by different users.
    """
    assert target_kind == "chapter", \
        f"prepare requires target_kind=chapter, got {target_kind}"
    assert ctx.inbox is not None, \
        "prepare requires ChapterInbox; build_inbox returned None"
    inbox = ctx.inbox
    chapter_id = target_id

    from typoon.sources.upload import UnpackError, unpack_zip
    from typoon.stages.prepare_archive import prepare_chapter_to_archive
    from typoon.sources.local import LocalSource

    chapter = await ctx.db.get_chapter(chapter_id)
    if chapter is None:
        logger.warning("prepare: chapter %d vanished, skipping", chapter_id)
        return

    handle = await ctx.db.get_inbox_handle(chapter_id)
    if handle is None:
        raise RuntimeError(
            f"prepare claim {chapter_id}: missing inbox handle "
            "(was the chapter created without an upload init?)",
        )

    with tempfile.TemporaryDirectory(prefix="typoon-prepare-") as tmp_str:
        tmp = Path(tmp_str)
        pages_dir = tmp / "pages"

        # complete_multipart is idempotent: if a prior attempt already
        # completed it, S3 returns NoSuchUpload and we proceed to fetch.
        try:
            await inbox.complete_multipart(
                tmp_id=handle.tmp_id,
                upload_id=handle.upload_id,
                parts=list(handle.parts),
            )
        except Exception as e:
            if "NoSuchUpload" not in str(e):
                raise

        zip_path = tmp / "chapter.zip"
        size = await inbox.fetch(
            tmp_id=handle.tmp_id, dest=zip_path,
        )
        if size <= 0:
            raise RuntimeError(f"prepare {chapter_id}: empty zip from inbox")
        try:
            n_unpacked = unpack_zip(zip_path, pages_dir)
        except UnpackError as e:
            raise RuntimeError(
                f"prepare {chapter_id}: unpack failed: {e}",
            ) from e
        if n_unpacked == 0:
            raise RuntimeError(f"prepare {chapter_id}: zip contained no pages")

        key, page_count = await prepare_chapter_to_archive(
            LocalSource(pages_dir),
            chapter_id=chapter_id,
            store=ctx.stores.pipeline,
            strategy="auto",
            work=tmp,
        )

        # Hash the just-uploaded archive for CAS dedup on next upload.
        archive_local = tmp / "prepared.bnl"
        prepared_hash = _hash_file(archive_local) if archive_local.exists() else ""

    await ctx.db.set_chapter_prepared(
        chapter_id,
        prepared_hash=prepared_hash,
        prepared_backend=ctx.stores.pipeline.backend_name,
        prepared_locator=key,
        page_count=page_count,
    )

    # Fan out: every pending draft on this chapter advances to scan.
    # In practice prepare is enqueued because at least one spawn
    # arrived for the chapter; that spawn either picked a fresh draft
    # (we advance it) or merged with one (we still need scan because
    # scan output is empty). We always enqueue scan keyed by chapter.
    await ctx.db.advance_task(
        target_kind="chapter", target_id=chapter_id,
        completed_stage="prepare", next_stage="scan",
        next_target_kind="chapter", next_target_id=chapter_id,
    )
    await ctx.db.clear_inbox_handle(chapter_id)

    try:
        await inbox.delete(
            tmp_id=handle.tmp_id,
        )
    except Exception as e:
        logger.warning(
            "inbox cleanup failed (chapter=%d tmp=%s): %s",
            chapter_id, handle.tmp_id, e,
        )


async def _handle_scan(
    ctx: StageContext, target_kind: str, target_id: int,
) -> None:
    """Chapter-scoped OCR + geometry pass. Output is shared across
    every translation referencing the chapter."""
    assert target_kind == "chapter", \
        f"scan requires target_kind=chapter, got {target_kind}"
    assert ctx.runtime is not None, "scan requires VisionRuntime"
    chapter_id = target_id

    # Source language: pull from the first draft pending on this
    # chapter. Multiple drafts in different source langs is rare and
    # the scan output (OCR text) is what would differ; for now use
    # the most-recently-created draft to seed source_lang.
    chapter = await ctx.db.get_chapter(chapter_id)
    if chapter is None:
        logger.warning("scan: chapter %d vanished, skipping", chapter_id)
        return
    material = await ctx.db.get_material(chapter["material_id"])
    source_lang = (
        material.get("languages") or ["unknown"]
    )[0] if material else "unknown"

    pipeline = ctx.stores.pipeline
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(
            pipeline, prepared_key(chapter_id), tmp,
        ) as reader:
            prepared = reader.chapter()
            result = await asyncio.to_thread(
                scan_chapter, prepared, reader, ctx.runtime,
                source_lang=source_lang,
                chapter_id=chapter_id, hook=ctx.hook,
            )

        await ctx.db.save_geometry(chapter_id, result.geometry_records())
        await ctx.db.save_bubbles(chapter_id, result.bubble_records())

        masks_path = tmp / "masks.npz"
        result.masks.pack(masks_path)
        locator = await pipeline.put(masks_key(chapter_id), masks_path)
        await ctx.db.set_chapter_masks(
            chapter_id,
            masks_backend=pipeline.backend_name,
            masks_locator=locator,
        )

    # Fan out to every pending draft for this chapter.
    draft_ids = await ctx.db.pending_drafts_for_chapter(chapter_id)
    if not draft_ids:
        # No draft waiting on us — possible after a delete race. Drop
        # the scan task; subsequent spawns will re-enqueue.
        await ctx.db.complete_task("chapter", chapter_id, "scan")
        return
    await ctx.db.complete_task("chapter", chapter_id, "scan")
    for did in draft_ids:
        await ctx.db.enqueue_task(
            target_kind="draft", target_id=did, stage="translate",
        )


async def _handle_translate(
    ctx: StageContext, target_kind: str, target_id: int,
) -> None:
    """Per-draft LLM pass. Reads chapter-scoped scan output; writes
    draft_briefs + translation_draft_bubbles bound to the draft."""
    assert target_kind == "draft", \
        f"translate requires target_kind=draft, got {target_kind}"
    draft_id = target_id

    draft = await ctx.db.get_draft(draft_id)
    if draft is None:
        logger.warning("translate: draft %d vanished, skipping", draft_id)
        return
    chapter_id = draft["chapter_id"]
    chapter = await ctx.db.get_chapter(chapter_id)
    if chapter is None:
        await ctx.db.update_draft_state(
            draft_id, state="error", error="chapter deleted",
        )
        return
    material = await ctx.db.get_material(chapter["material_id"])

    await ctx.db.update_draft_state(draft_id, state="running")

    tctx = make_ctx(
        chapter_id=chapter_id,
        draft_id=draft_id,
        chapter_position=chapter["position"],
        material_id=chapter["material_id"],
        owner_id=draft["created_by"],
        source_lang=draft["source_lang"],
        target_lang=draft["target_lang"],
        store=ctx.db,
        config=ctx.config,
        hook=ctx.hook,
    )

    pipeline = ctx.stores.pipeline
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(
            pipeline, prepared_key(chapter_id), tmp,
        ) as reader:
            scanned = await load_scanned(reader, ctx.db, chapter_id)
            translated, brief = await translate_chapter(scanned, reader, tctx)

    await ctx.db.save_draft_brief(draft_id, brief.to_dict())
    await ctx.db.save_draft_bubbles(draft_id, translated.to_db_records())
    await ctx.db.update_draft_state(draft_id, state="done")

    # Append into the draft owner's translator_memory so subsequent
    # chapters in the same (user, material, target_lang) inherit the
    # context. The memory row is created lazily by the API/UI when the
    # user first touches the material; we only append a brief snapshot
    # here. If the row does not exist yet (e.g. background cache job),
    # we skip the append silently.
    #
    # Cache-reuse note: when another user later hits this draft via
    # the visibility cache, this append fires once for the original
    # spawner. Cache consumers don't accumulate memory because their
    # spawn never ran the agent loop. That's deliberate — memory
    # tracks who DID the translation, not who READ it.
    if material is not None:
        memory_row = await ctx.db.get_translator_memory(
            user_id=draft["created_by"],
            material_id=chapter["material_id"],
            target_lang=draft["target_lang"],
        )
        if memory_row is None:
            # Lazy create: first translate of this (user, material,
            # target_lang) seeds the memory row so subsequent chapters
            # have somewhere to append briefs.
            memory_row = await ctx.db.upsert_translator_memory(
                user_id=draft["created_by"],
                material_id=chapter["material_id"],
                source_lang=draft["source_lang"],
                target_lang=draft["target_lang"],
            )
        await ctx.db.append_memory_brief(
            memory_id=memory_row["id"],
            chapter_id=chapter_id,
            brief_json=brief.to_dict(),
            summary=brief.summary_line(),
        )

    # Fan out render: target the draft (shared default render). When
    # users later add sparse edits, individual translations re-render
    # to their own t/{id}/render.bnl key.
    await ctx.db.advance_task(
        target_kind="draft", target_id=draft_id,
        completed_stage="translate", next_stage="render",
        next_target_kind="draft", next_target_id=draft_id,
    )


async def _handle_render(
    ctx: StageContext, target_kind: str, target_id: int,
) -> None:
    """Render runs either against a draft (default shared archive) or
    against a single translation (per-user fork when sparse edits
    diverge). The target kind dictates which DB pointer is set.
    """
    assert ctx.runtime is not None, "render requires VisionRuntime"

    if target_kind == "draft":
        draft_id       = target_id
        translation_id = None
        draft = await ctx.db.get_draft(draft_id)
        if draft is None:
            logger.warning("render: draft %d vanished, skipping", draft_id)
            return
        chapter_id = draft["chapter_id"]
    elif target_kind == "translation":
        translation_id = target_id
        t = await ctx.db.get_translation(translation_id)
        if t is None:
            logger.warning(
                "render: translation %d vanished, skipping", translation_id,
            )
            return
        draft_id = t.get("draft_id")
        if not draft_id:
            await ctx.db.fail_task(
                target_kind, target_id, "render",
                "translation has no draft_id (cannot render bubbles)",
            )
            return
        d = await ctx.db.get_draft(int(draft_id))
        if d is None:
            await ctx.db.fail_task(
                target_kind, target_id, "render",
                f"translation draft {draft_id} missing",
            )
            return
        chapter_id = int(d["chapter_id"])
    else:
        raise AssertionError(
            f"render target_kind must be draft|translation, got {target_kind}",
        )

    pipeline = ctx.stores.pipeline
    public   = ctx.stores.public

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with await open_prepared_reader(
            pipeline, prepared_key(chapter_id), tmp,
        ) as reader:
            translated, page_geoms = await load_translated_with_geometry(
                reader, ctx.db, chapter_id, draft_id,
                translation_id=translation_id,
            )

            masks_local = tmp / "masks.npz"
            await pipeline.get(masks_key(chapter_id), masks_local)
            masks = MaskStore.unpack(masks_local)

            # Pages flagged as full-page noise by the brief drop out
            # of the public archive. Brief is present at render time
            # because translate is a hard prerequisite.
            brief_dict = await ctx.db.get_draft_brief(draft_id)
            skip_pages: frozenset[int] = frozenset(
                int(p) for p in (brief_dict or {}).get("noise_pages", [])
            )

            locator, _public_page_count = await render_chapter_to_archive(
                translated,
                target_kind=target_kind,
                target_id=target_id,
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

    if target_kind == "draft":
        # Default draft render — every translation pointing at this
        # draft with no edits will read this archive. Persist on the
        # draft row so the translate route can serve it.
        await ctx.db.update_draft_archive(
            draft_id,
            archive_backend=public.backend_name,
            archive_locator=locator,
        )
    else:
        await ctx.db.update_translation_archive(
            translation_id,
            archive_backend=public.backend_name,
            archive_locator=locator,
        )

    await ctx.db.complete_task(target_kind, target_id, "render")


_HANDLERS: dict[str, StageHandler] = {
    "prepare":   _handle_prepare,
    "scan":      _handle_scan,
    "translate": _handle_translate,
    "render":    _handle_render,
}


# ── Stage pump ─────────────────────────────────────────────────────


async def _run_one(
    ctx:    StageContext,
    stage:  str,
    target_kind: str,
    target_id:   int,
) -> None:
    """Dispatch one claim through its handler with uniform lifecycle.

    Emits StageStarted/StageDone/StageFailed centrally so adding a new
    stage means one handler function + one entry in _HANDLERS.
    """
    chapter_id = await _chapter_id_for_target(ctx.db, target_kind, target_id)
    if chapter_id is None:
        # Target vanished between claim and dispatch.
        return

    draft_id       = target_id if target_kind == "draft" else 0
    translation_id = target_id if target_kind == "translation" else 0

    ctx.hook.on(StageStarted(
        chapter_id=chapter_id,
        draft_id=draft_id, translation_id=translation_id,
        stage=stage,
    ))
    try:
        await _HANDLERS[stage](ctx, target_kind, target_id)
        ctx.hook.on(StageDone(
            chapter_id=chapter_id,
            draft_id=draft_id, translation_id=translation_id,
            stage=stage,
        ))
    except (TransientCredentialError, UpstreamUnavailable) as e:
        logger.warning(
            "%s requeue %s:%d (%s): %s",
            stage, target_kind, target_id, type(e).__name__, e,
        )
        await ctx.db.requeue_task(target_kind, target_id, stage, str(e))
        ctx.hook.on(StageFailed(
            chapter_id=chapter_id,
            draft_id=draft_id, translation_id=translation_id,
            stage=stage, error=e,
        ))
        await asyncio.sleep(_REQUEUE_BACKOFF_SECONDS)
    except Exception as e:
        logger.exception("%s failed %s:%d", stage, target_kind, target_id)
        await ctx.db.fail_task(target_kind, target_id, stage, str(e))
        ctx.hook.on(StageFailed(
            chapter_id=chapter_id,
            draft_id=draft_id, translation_id=translation_id,
            stage=stage, error=e,
        ))


async def _stage_pump(
    ctx:    StageContext,
    stage:  str,
    dsn:    str,
    stop:   asyncio.Event,
) -> None:
    """Run claims for one stage until cancelled.

    Wake-up sources:
      1. NOTIFY on `typoon_task_<stage>` — fires ~5ms after insert/release.
      2. 30s safety poll — covers missed notifications.
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

        while not stop.is_set():
            claimed_any = True
            while claimed_any and not stop.is_set():
                claimed_any = False
                while not stop.is_set():
                    claimed = await ctx.db.claim_task(stage, worker_id)
                    if claimed is None:
                        break
                    target_kind, target_id = claimed
                    claimed_any = True
                    await _run_one(ctx, stage, target_kind, target_id)
            if stop.is_set():
                break

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
    """Start every stage pump for `role` and block until shutdown."""
    from typoon.config import load_config

    config, paths = load_config() if config is None else (config, Paths())
    paths.ensure()

    db    = await PostgresStore.open(
        config.database_url,
        pool_min_size=config.database.pool_min_size,
        pool_max_size=config.database.pool_max_size,
        statement_cache_size=config.database.statement_cache_size,
    )
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

    released = await db.release_claims_by_prefix(_host_prefix())
    if released:
        logger.info(
            "released %d ghost claim(s) from prior PID(s) on this host",
            released,
        )

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    try:
        async with asyncio.TaskGroup() as tg:
            for stage in stages:
                pumps = translate_concurrency if stage == "translate" else 1
                for _ in range(pumps):
                    tg.create_task(
                        _stage_pump(ctx, stage, config.database_url, stop),
                    )
            await stop.wait()
    finally:
        await stores.aclose()
        await bus.close()
        await db.close()
