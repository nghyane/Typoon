"""3-stage queue pipeline: feeder → scanner → translator_and_renderer."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from ...events import ChapterDone, ChapterSkipped, ChapterStart, Hook, PipelineError, SeriesProgress
from ....ports import Store

from .chapter import _do_render_only, _do_translate_and_render
from .job import _Job
from .policy import ResumePolicy, _decide


async def _iter_chapters(chapters, stream):
    if chapters:
        for item in chapters:
            yield item
    if stream:
        while True:
            item = await stream.get()
            if item is None:
                break
            yield item


async def run_pipeline(
    engine,
    store: Store,
    config,
    project_id: int,
    chapters,
    hook: Hook,
    on_chapter,
    policy: ResumePolicy,
    chapter_stream: asyncio.Queue | None,
    total_hint: int,
) -> dict:
    """Run the full pipeline for a project. Returns {done, failed, skipped}."""
    from collections.abc import Callable

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
        async for ch_num, source in _iter_chapters(chapters, chapter_stream):
            if h.quit_requested:
                break
            status = await store.get_chapter_status(project_id, ch_num)
            retry_count = await store.get_chapter_retry_count(project_id, ch_num)
            action = _decide(status, pol, retry_count)
            if action == "skip":
                counts["skipped"] += 1
                h.on(ChapterSkipped(chapter=ch_num, reason=status or "pending"))
                h.on(SeriesProgress(done=counts["done"], failed=counts["failed"],
                                    skipped=counts["skipped"], total=total))
                continue
            await store.add_chapter(project_id, ch_num)
            if action == "clean":
                await store.delete_chapter_data(project_id, ch_num)
            await scan_q.put(_Job(chapter=ch_num, source=source,
                                  project_id=project_id, t0=time.time(), action=action))
        await scan_q.put(None)

    async def scanner():
        loop = asyncio.get_running_loop()
        while True:
            job = await scan_q.get()
            if job is None:
                await render_q.put(None)
                return
            try:
                h.on(ChapterStart(
                    project_id=project_id,
                    chapter=job.chapter,
                    pages=job.source.page_count(),
                ))
                job.pages, job.images = await loop.run_in_executor(
                    gpu, engine.preprocess, job.source, h)
            except Exception as e:
                await store.set_chapter_status(project_id, job.chapter, "failed")
                await store.increment_retry_count(project_id, job.chapter)
                h.on(PipelineError(stage=f"scan ch{job.chapter}", error=e))
                counts["failed"] += 1
                h.on(SeriesProgress(done=counts["done"], failed=counts["failed"],
                                    skipped=counts["skipped"], total=total))
                continue
            await render_q.put(job)

    async def translator_and_renderer():
        loop = asyncio.get_running_loop()
        while True:
            job = await render_q.get()
            if job is None:
                return
            try:
                if job.action == "render":
                    await _do_render_only(job, store, engine, config, h, gpu)
                else:
                    await _do_translate_and_render(job, store, engine, config, h, loop, gpu)

                n_bubbles = sum(len(p.bubbles) for p in job.pages)
                h.on(ChapterDone(chapter=job.chapter, pages=len(job.pages),
                                 bubbles=n_bubbles, elapsed=time.time() - job.t0))
                counts["done"] += 1
                if on_chapter and job.pages:
                    try:
                        on_chapter(job.chapter, job.pages)
                    except Exception as e:
                        h.on(PipelineError(stage=f"output ch{job.chapter}", error=e))
            except Exception as e:
                await store.set_chapter_status(project_id, job.chapter, "failed")
                await store.increment_retry_count(project_id, job.chapter)
                h.on(PipelineError(stage=f"chapter {job.chapter}", error=e))
                counts["failed"] += 1
            h.on(SeriesProgress(done=counts["done"], failed=counts["failed"],
                                skipped=counts["skipped"], total=total))

    try:
        await asyncio.gather(feeder(), scanner(), translator_and_renderer())
    finally:
        engine.unload_scan_models(h)
        engine.unload_erase_models(h)
        gpu.shutdown(wait=False)

    return counts
