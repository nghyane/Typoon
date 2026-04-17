"""Runner — chapter processing phases.

Stateless pipeline: preprocess → translate → render → consolidate.
No status tracking, no skip/resume — that's Orchestrator's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import Config
from .events import (
    ChapterStart,
    Hook,
    KnowledgeDone,
    KnowledgeStart,
    PipelineError,
    TranslateDone,
    TranslateStart,
)
from .ports import ChapterSource, Store
from .providers import make_context_provider, make_translation_provider
from .types import Page, Session
from .vision.chapter_images import ChapterImages

if TYPE_CHECKING:
    from .engine import Engine


class ChapterRunner:
    """Executes pipeline phases for a single chapter. No DB status tracking."""

    def __init__(self, engine: Engine, store: Store, config: Config) -> None:
        self.engine = engine
        self.store = store
        self.config = config

    # ── Phases ────────────────────────────────────────────────────

    async def scan_and_translate(
        self, project_id: int, chapter: float,
        source: ChapterSource, hook: Hook,
    ) -> tuple[list[Page], ChapterImages]:
        """Phase 1+2: preprocess → translate. Returns (pages, images)."""
        pages, images = await self._preprocess(project_id, chapter, source, hook)

        session = await self._session(project_id, chapter, images, hook)
        from .translation.agent import translate_pages

        total = sum(len(p.bubbles) for p in pages)
        hook.on(TranslateStart(total_bubbles=total))
        turns, error = await translate_pages(pages, session)
        translated = sum(1 for p in pages for b in p.bubbles if b.translated_text is not None)
        hook.on(TranslateDone(translated=translated, total=total, turns=turns))
        if error:
            raise error

        _check_completeness(pages, chapter, hook)
        return pages, images

    async def scan_and_load(
        self, project_id: int, chapter: float,
        source: ChapterSource, hook: Hook,
    ) -> tuple[list[Page], ChapterImages]:
        """Phase 1 + load: preprocess → load translations from DB."""
        pages, images = await self._preprocess(project_id, chapter, source, hook)

        existing = await self.store.get_chapter_translations(project_id, chapter)
        trans_map = {f"p{t['page']}_b{t['idx']}": t["translated_text"] for t in existing}
        for p in pages:
            for b in p.bubbles:
                if b.id in trans_map:
                    b.translated_text = trans_map[b.id]
        return pages, images

    def render(self, pages: list[Page], images: ChapterImages, hook: Hook) -> None:
        """Phase 3: erase + render translated text onto pages."""
        self.engine.erase_and_render(pages, images, hook)

    async def consolidate(
        self, project_id: int, chapter: float,
        pairs: list[tuple[str, str]], hook: Hook,
    ) -> None:
        """Phase 4: extract knowledge from translation pairs."""
        from .translation.knowledge import consolidate

        if not pairs:
            return

        session = await self._session(project_id, chapter, None, hook)
        hook.on(KnowledgeStart(chapter=chapter, pairs=len(pairs)))
        result = await consolidate(session, chapter, pairs)
        hook.on(KnowledgeDone(chapter=chapter, turns=result.turns))
        if result.error:
            hook.on(PipelineError(stage="knowledge", error=result.error))

    def unload_models(self, hook: Hook) -> None:
        self.engine.unload_scan_models(hook)
        self.engine.unload_erase_models(hook)

    # ── Internal ─────────────────────────────────────────────────

    async def _preprocess(
        self, project_id: int, chapter: float,
        source: ChapterSource, hook: Hook,
    ) -> tuple[list[Page], ChapterImages]:
        await source.fetch()
        hook.on(ChapterStart(project_id=project_id, chapter=chapter, pages=source.page_count()))
        return self.engine.preprocess(source, hook)

    async def _session(
        self, project_id: int, chapter: float,
        images: ChapterImages | None, hook: Hook,
    ) -> Session:
        project = await self.store.get_project(project_id)
        sl = project.get("source_lang", "en") if project else "en"
        tl = (project.get("target_lang", self.config.default_target_lang)
              if project else self.config.default_target_lang)
        try:
            ctx = make_context_provider(self.config)
        except (ValueError, KeyError):
            ctx = make_translation_provider(self.config)

        source = _PageImageSource(images) if images else None

        return Session(
            store=self.store, source=source, project_id=project_id,
            source_lang=sl, target_lang=tl,
            provider=make_translation_provider(self.config),
            context_provider=ctx, hook=hook,
            glossary=await self.store.get_glossary(project_id),
            knowledge=await self.store.get_knowledge(project_id, before_chapter=chapter),
        )


class _PageImageSource:
    """Exposes ChapterImages pages for view_page/view_bubble tools."""

    def __init__(self, images: ChapterImages) -> None:
        self._images = images

    def page_count(self) -> int:
        return self._images.page_count()

    def load_page(self, index: int) -> np.ndarray:
        if not self._images.alive:
            raise RuntimeError(f"Page {index} image already freed")
        return self._images.page(index)


def _check_completeness(pages: list[Page], chapter: float, hook: Hook) -> None:
    all_bubbles = [b for p in pages for b in p.bubbles]
    if not all_bubbles:
        return
    translated = sum(1 for b in all_bubbles if b.translated_text is not None)
    ratio = translated / len(all_bubbles)
    if ratio < 0.8:
        raise RuntimeError(f"Too few translations: {translated}/{len(all_bubbles)} (<80%)")
    if translated < len(all_bubbles):
        hook.on(PipelineError(
            stage=f"chapter {chapter}",
            error=RuntimeError(f"{len(all_bubbles) - translated} bubbles untranslated"),
        ))
