"""Per-chapter processing: translate+render or render-only, plus knowledge consolidation."""

from __future__ import annotations

import asyncio

from ....domain.bubble import Page, Session
from ...events import Hook, KnowledgeDone, KnowledgeStart, PipelineError, TranslateDone, TranslateStart, TranslationReady


async def _do_translate_and_render(job, store, engine, config, hook, loop, gpu):
    await store.set_chapter_status(job.project_id, job.chapter, "translating")

    session = await _make_session(store, config, job.project_id, job.chapter, job.images, hook)
    from ....translation.translate import translate_pages

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
    hook.on(TranslationReady(
        pages=len(job.pages),
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
        from ....translation.knowledge import consolidate
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
    from ....providers import make_context_provider, make_translation_provider

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


def _check_completeness(pages: list[Page], hook: Hook) -> None:
    all_bubbles = [b for p in pages for b in p.bubbles]
    if not all_bubbles:
        return
    translated = sum(1 for b in all_bubbles if b.translated_text is not None)
    ratio = translated / len(all_bubbles)
    if ratio < 0.8:
        raise RuntimeError(f"Too few translations: {translated}/{len(all_bubbles)} (<80%)")
    if translated < len(all_bubbles):
        hook.on(PipelineError(
            stage="translate",
            error=RuntimeError(f"{len(all_bubbles) - translated} bubbles untranslated"),
        ))
