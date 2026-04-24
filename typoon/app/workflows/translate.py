"""Translate workflow — run translation on scanned pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...events import Hook, PipelineError, TranslateDone, TranslateStart

if TYPE_CHECKING:
    from ...domain.bubble import Page, Session


async def translate_chapter(
    pages: list[Page],
    session: Session,
    hook: Hook,
) -> tuple[int, Exception | None]:
    """Phase 2: translate all bubbles. Returns (turns, error)."""
    from ...translation.agent import translate_pages

    total = sum(len(p.bubbles) for p in pages)
    hook.on(TranslateStart(total_bubbles=total))
    turns, error = await translate_pages(pages, session)
    translated = sum(1 for p in pages for b in p.bubbles if b.translated_text is not None)
    hook.on(TranslateDone(translated=translated, total=total, turns=turns))

    if error:
        raise error

    _check_completeness(pages, hook)
    return turns, error


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
