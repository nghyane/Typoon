"""Orchestrator — project-level chapter lifecycle management.

3-stage queue pipeline:

    Scanner worker ──→ [scan_q] ──→ Translator worker ──→ [render_q] ──→ Renderer worker

Translate is sequential (needs knowledge context from prior chapters).
Scan and render overlap with translate via bounded async queues.
Backpressure: maxsize=1 → at most 2 chapters in memory.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .events import (
    ChapterDone,
    ChapterSkipped,
    Hook,
    PipelineError,
    SeriesProgress,
    TranslationReady,
)
from .ports import ChapterSource, Store
from .runner import ChapterRunner
from .types import Page
from .vision.chapter_images import ChapterImages


# ── Policy ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResumePolicy:
    """Controls how the orchestrator handles each chapter state.

    Presets:
        CLI:      ResumePolicy()
        Desktop:  ResumePolicy(resume_translated=False, retry_failed=False)
        Platform: ResumePolicy(max_retries=3)
        Force:    ResumePolicy(force=True)
    """
    force: bool = False
    resume_translated: bool = True
    retry_failed: bool = True
    max_retries: int = 0  # 0 = unlimited


def _decide(status: str | None, policy: ResumePolicy, retry_count: int = 0) -> str:
    """Returns: "skip", "translate", "render", or "clean"."""
    match status:
        case "done":
            return "clean" if policy.force else "skip"
        case "translated":
            return "render" if policy.resume_translated else "skip"
        case "rendering":
            return "render"
        case "translating":
            return "clean"
        case "failed":
            if not policy.retry_failed:
                return "skip"
            if policy.max_retries > 0 and retry_count >= policy.max_retries:
                return "skip"
            return "clean"
        case _:
            return "translate"


# ── Job ──────────────────────────────────────────────────────────


@dataclass
class _Job:
    """Flows through the pipeline. Each phase fills its part."""

    chapter: float
    source: ChapterSource
    project_id: int
    t0: float
    action: str  # "translate" | "render"
    pages: list[Page] | None = None
    images: ChapterImages | None = None
    pairs: list[tuple[str, str]] | None = None


_SENTINEL = None  # Queue terminator


# ── Orchestrator ─────────────────────────────────────────────────


class Orchestrator:
    """Queue-based pipeline: scan → translate → render.

    Each phase runs as an async worker connected by bounded queues.
    No shared mutable state between phases.
    """

    def __init__(self, runner: ChapterRunner, store: Store) -> None:
        self.runner = runner
        self.store = store

    async def run(
        self,
        project_id: int,
        chapters: list[tuple[float, ChapterSource]] | None = None,
        hook: Hook | None = None,
        on_chapter: Callable[[float, list[Page]], None] | None = None,
        policy: ResumePolicy | None = None,
        chapter_stream: asyncio.Queue | None = None,
        total_hint: int = 0,
    ) -> dict:
        if not chapters and not chapter_stream:
            return {"done": 0, "failed": 0, "skipped": 0}

        pol = policy or ResumePolicy()
        h = hook or Hook()
        total = len(chapters) if chapters else total_hint
        counts = {"done": 0, "failed": 0, "skipped": 0}

        scan_q: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=1)
        render_q: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=1)
        gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

        async def feeder():
            """Filter chapters by policy, push actionable ones to scan_q."""
            async for ch_num, source in self._iter_chapters(chapters, chapter_stream):
                if h.quit_requested:
                    break

                status = await self.store.get_chapter_status(project_id, ch_num)
                retry_count = await self.store.get_chapter_retry_count(project_id, ch_num)
                action = _decide(status, pol, retry_count)

                if action == "skip":
                    counts["skipped"] += 1
                    h.on(ChapterSkipped(chapter=ch_num, reason=status or "pending"))
                    h.on(SeriesProgress(
                        done=counts["done"], failed=counts["failed"],
                        skipped=counts["skipped"], total=total,
                    ))
                    continue

                await self.store.add_chapter(project_id, ch_num)
                if action == "clean":
                    await self.store.delete_chapter_data(project_id, ch_num)

                job = _Job(
                    chapter=ch_num, source=source,
                    project_id=project_id, t0=time.time(),
                    action=action,
                )
                await scan_q.put(job)

            await scan_q.put(_SENTINEL)

        async def scanner():
            """Preprocess chapters (GPU bound → single-thread executor)."""
            loop = asyncio.get_running_loop()
            while True:
                job = await scan_q.get()
                if job is _SENTINEL:
                    await render_q.put(_SENTINEL)
                    return

                try:
                    job.pages, job.images = await loop.run_in_executor(
                        gpu, self.runner.engine.preprocess, job.source, h,
                    )
                except Exception as e:
                    await self.store.set_chapter_status(project_id, job.chapter, "failed")
                    await self.store.increment_retry_count(project_id, job.chapter)
                    h.on(PipelineError(stage=f"scan ch{job.chapter}", error=e))
                    counts["failed"] += 1
                    h.on(SeriesProgress(
                        done=counts["done"], failed=counts["failed"],
                        skipped=counts["skipped"], total=total,
                    ))
                    continue

                await render_q.put(job)

        async def translator_and_renderer():
            """Translate (sequential, IO bound) then render (CPU bound)."""
            loop = asyncio.get_running_loop()
            while True:
                job = await render_q.get()
                if job is _SENTINEL:
                    return

                try:
                    if job.action == "render":
                        await self._do_render_only(job, h, gpu)
                    else:
                        await self._do_translate_and_render(job, h, loop, gpu)

                    n_bubbles = sum(len(p.bubbles) for p in job.pages)
                    h.on(ChapterDone(
                        chapter=job.chapter, pages=len(job.pages),
                        bubbles=n_bubbles, elapsed=time.time() - job.t0,
                    ))
                    counts["done"] += 1

                    if on_chapter and job.pages:
                        try:
                            on_chapter(job.chapter, job.pages)
                        except Exception as e:
                            h.on(PipelineError(stage=f"output ch{job.chapter}", error=e))

                except Exception as e:
                    await self.store.set_chapter_status(project_id, job.chapter, "failed")
                    await self.store.increment_retry_count(project_id, job.chapter)
                    h.on(PipelineError(stage=f"chapter {job.chapter}", error=e))
                    counts["failed"] += 1

                h.on(SeriesProgress(
                    done=counts["done"], failed=counts["failed"],
                    skipped=counts["skipped"], total=total,
                ))

        try:
            await asyncio.gather(feeder(), scanner(), translator_and_renderer())
        finally:
            self.runner.unload_models(h)
            gpu.shutdown(wait=False)

        return counts

    # ── Pipeline stages ──────────────────────────────────────────

    async def _do_translate_and_render(
        self, job: _Job, hook: Hook, loop: asyncio.AbstractEventLoop,
        gpu: ThreadPoolExecutor,
    ) -> None:
        """Translate → consolidate (fire-and-forget) → render."""
        await self.store.set_chapter_status(job.project_id, job.chapter, "translating")

        # Translate
        session = await self.runner._session(job.project_id, job.chapter, job.images, hook)
        from .translation.agent import translate_pages
        from .runner import _check_completeness

        total = sum(len(p.bubbles) for p in job.pages)
        hook.on(TranslateStart(total_bubbles=total))
        turns, error = await translate_pages(job.pages, session)
        translated = sum(1 for p in job.pages for b in p.bubbles if b.translated_text is not None)
        hook.on(TranslateDone(translated=translated, total=total, turns=turns))
        if error:
            raise error
        _check_completeness(job.pages, job.chapter, hook)

        # Save translations
        all_bubbles = [b for p in job.pages for b in p.bubbles]
        await self.store.save_translations(job.project_id, job.chapter, all_bubbles)
        await self.store.set_chapter_status(job.project_id, job.chapter, "translated")
        hook.on(TranslationReady(
            pages=len(job.pages),
            translated=sum(1 for b in all_bubbles if b.translated_text is not None),
            total=len(all_bubbles),
        ))

        # Extract pairs BEFORE render (consolidate is decoupled)
        job.pairs = [
            (b.source_text, b.translated_text or "")
            for b in all_bubbles if b.translated_text is not None
        ]
        asyncio.create_task(
            self.runner.consolidate(job.project_id, job.chapter, job.pairs, hook)
        )

        # Render (GPU bound → same single-thread executor)
        await self.store.set_chapter_status(job.project_id, job.chapter, "rendering")
        await loop.run_in_executor(gpu, self.runner.render, job.pages, job.images, hook)
        await self.store.set_chapter_status(job.project_id, job.chapter, "done")

    async def _do_render_only(self, job: _Job, hook: Hook, gpu: ThreadPoolExecutor) -> None:
        """Load translations from DB → consolidate → render."""
        await self.store.set_chapter_status(job.project_id, job.chapter, "rendering")

        existing = await self.store.get_chapter_translations(job.project_id, job.chapter)
        trans_map = {f"p{t['page']}_b{t['idx']}": t["translated_text"] for t in existing}
        for p in job.pages:
            for b in p.bubbles:
                if b.id in trans_map:
                    b.translated_text = trans_map[b.id]

        pairs = [
            (b.source_text, b.translated_text or "")
            for p in job.pages for b in p.bubbles
            if b.translated_text is not None
        ]
        await self.runner.consolidate(job.project_id, job.chapter, pairs, hook)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(gpu, self.runner.render, job.pages, job.images, hook)
        await self.store.set_chapter_status(job.project_id, job.chapter, "done")

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    async def _iter_chapters(chapters, stream):
        """Yield from static list first, then from async queue (sentinel=None)."""
        if chapters:
            for item in chapters:
                yield item
        if stream:
            while True:
                item = await stream.get()
                if item is None:
                    break
                yield item
