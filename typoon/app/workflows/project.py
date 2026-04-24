"""Project workflow — multi-chapter orchestration.

3-stage queue pipeline:
    feeder → [scan_q] → scanner → [render_q] → translator_and_renderer

Preserves the existing Orchestrator logic but as a plain async function.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ...events import (
    ChapterDone,
    ChapterSkipped,
    Hook,
    PipelineError,
    SeriesProgress,
    TranslateDone,
    TranslateStart,
    TranslationReady,
)
from ...ports import ChapterSource, Store
from ...domain.bubble import Page
from ...vision.chapter_images import ChapterImages


@dataclass(frozen=True)
class ResumePolicy:
    force: bool = False
    resume_translated: bool = True
    retry_failed: bool = True
    max_retries: int = 0


def _decide(status: str | None, policy: ResumePolicy, retry_count: int = 0) -> str:
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


@dataclass
class _Job:
    chapter: float
    source: ChapterSource
    project_id: int
    t0: float
    action: str
    pages: list[Page] | None = None
    images: ChapterImages | None = None
    pairs: list[tuple[str, str]] | None = None


async def translate_project(
    engine,
    store: Store,
    config,
    project_id: int,
    chapters: list[tuple[float, ChapterSource]] | None = None,
    hook: Hook | None = None,
    on_chapter: Callable[[float, list[Page]], None] | None = None,
    policy: ResumePolicy | None = None,
    chapter_stream: asyncio.Queue | None = None,
    total_hint: int = 0,
) -> dict:
    """Run the full pipeline for a project. Returns {done, failed, skipped}."""
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


async def _do_translate_and_render(job, store, engine, config, hook, loop, gpu):
    await store.set_chapter_status(job.project_id, job.chapter, "translating")

    session = await _make_session(store, config, job.project_id, job.chapter, job.images, hook)
    from ...translation.agent import translate_pages
    from .translate import _check_completeness

    total = sum(len(p.bubbles) for p in job.pages)
    hook.on(TranslateStart(total_bubbles=total))
    turns, error = await translate_pages(job.pages, session)
    translated = sum(1 for p in job.pages for b in p.bubbles if b.translated_text is not None)
    hook.on(TranslateDone(translated=translated, total=total, turns=turns))
    if error:
        raise error
    _check_completeness(job.pages, hook)

    all_bubbles = [b for p in job.pages for b in p.bubbles]
    await store.save_translations(job.project_id, job.chapter, all_bubbles)
    await store.set_chapter_status(job.project_id, job.chapter, "translated")
    hook.on(TranslationReady(pages=len(job.pages),
                             translated=sum(1 for b in all_bubbles if b.translated_text is not None),
                             total=len(all_bubbles)))

    job.pairs = [(b.source_text, b.translated_text or "") for b in all_bubbles if b.translated_text is not None]
    asyncio.create_task(_consolidate(store, config, job.project_id, job.chapter, job.pairs, hook))

    await store.set_chapter_status(job.project_id, job.chapter, "rendering")
    await loop.run_in_executor(gpu, engine.erase_and_render, job.pages, job.images, hook)
    await store.set_chapter_status(job.project_id, job.chapter, "done")


async def _do_render_only(job, store, engine, config, hook, gpu):
    await store.set_chapter_status(job.project_id, job.chapter, "rendering")

    existing = await store.get_chapter_translations(job.project_id, job.chapter)
    trans_map = {f"p{t['page']}_b{t['idx']}": t["translated_text"] for t in existing}
    for p in job.pages:
        for b in p.bubbles:
            if b.id in trans_map:
                b.translated_text = trans_map[b.id]

    pairs = [(b.source_text, b.translated_text or "")
             for p in job.pages for b in p.bubbles if b.translated_text is not None]
    await _consolidate(store, config, job.project_id, job.chapter, pairs, hook)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu, engine.erase_and_render, job.pages, job.images, hook)
    await store.set_chapter_status(job.project_id, job.chapter, "done")


async def _consolidate(store, config, project_id, chapter, pairs, hook):
    if not pairs:
        return
    try:
        from ...translation.knowledge import consolidate
        from ...events import KnowledgeDone, KnowledgeStart
        session = await _make_session(store, config, project_id, chapter, None, hook)
        hook.on(KnowledgeStart(chapter=chapter, pairs=len(pairs)))
        result = await consolidate(session, chapter, pairs)
        hook.on(KnowledgeDone(chapter=chapter, turns=result.turns))
        if result.error:
            hook.on(PipelineError(stage="knowledge", error=result.error))
    except Exception as e:
        hook.on(PipelineError(stage="knowledge", error=e))


async def _make_session(store, config, project_id, chapter, images, hook):
    """Build a Session for translation/knowledge agents."""
    from ...providers import make_context_provider, make_translation_provider
    from ...domain.bubble import Session

    project = await store.get_project(project_id)
    sl = project.get("source_lang", "en") if project else "en"
    tl = (project.get("target_lang", config.default_target_lang)
          if project else config.default_target_lang)
    try:
        ctx = make_context_provider(config)
    except (ValueError, KeyError):
        ctx = make_translation_provider(config)

    source = _PageImageSource(images) if images else None

    return Session(
        store=store, source=source, project_id=project_id,
        source_lang=sl, target_lang=tl,
        provider=make_translation_provider(config),
        context_provider=ctx, hook=hook,
        glossary=await store.get_glossary(project_id),
        knowledge=await store.get_knowledge(project_id, before_chapter=chapter),
    )


class _PageImageSource:
    def __init__(self, images):
        self._images = images

    def page_count(self) -> int:
        return self._images.page_count()

    def load_page(self, index: int):
        if not self._images.alive:
            raise RuntimeError(f"Page {index} image already freed")
        return self._images.page(index)


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
