"""Translate stage — ScannedChapter → TranslatedChapter.

Bridges the new typed domain contracts with the existing translation
agents (context.py, page.py) which still operate on Bubble/Page.
The bridge is intentional and temporary: agents will be updated to
work with ScannedBubble/TranslatedBubble directly in a later pass.
"""

from __future__ import annotations

from typoon.adapters.session import Session
from typoon.domain.bubble import Bubble, Page
from typoon.domain.scan import ScannedBubble, ScannedChapter, ScannedPage
from typoon.domain.translate import TranslatedBubble, TranslatedChapter, TranslatedPage
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import PipelineError
from typoon.translation.context import build_chapter_brief
from typoon.translation.keys import assign_keys
from typoon.translation.page import translate_window
from typoon.translation.tools.submit import TextKind

_PAGE_WINDOW_MAX_KEYS = 25


async def translate_chapter(
    scanned: ScannedChapter,
    session: Session,
    *,
    artifacts: ArtifactSink | None = None,
) -> TranslatedChapter:
    """Translate a ScannedChapter and return a TranslatedChapter.

    Raises on unrecoverable agent error.
    """
    # Build legacy Bubble/Page objects for existing agents.
    bubble_map, pages = _build_legacy(scanned)

    all_bubbles = [b for p in pages for b in p.bubbles]
    if not all_bubbles:
        return _empty_result(scanned)

    key_map = assign_keys(all_bubbles, project_id=session.project_id, chapter=session.chapter)

    try:
        brief, _ = await build_chapter_brief(pages, session, key_map)

        for window in _page_windows(pages):
            accepted, _ = await translate_window(
                session, brief=brief, bubbles=window,
                key_map=key_map, all_pages=pages,
            )
            for op in accepted:
                b = key_map[op.key]
                b.translation_status = op.kind
                b.translated_text = op.text if op.kind != TextKind.skip.value else ""

        await session.store.save_chapter_brief(
            session.project_id, session.chapter, brief.to_dict()
        )
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        raise

    return _build_result(scanned, pages, bubble_map)


# ── Bridge helpers ────────────────────────────────────────────────────


def _build_legacy(
    scanned: ScannedChapter,
) -> tuple[dict[int, ScannedBubble], list[Page]]:
    """Convert ScannedChapter to Bubble/Page for legacy agents.

    Returns (id_to_scanned_bubble map, legacy pages).
    The id() of each Bubble maps back to its ScannedBubble.
    """
    bubble_map: dict[int, ScannedBubble] = {}
    pages: list[Page] = []
    for sp in scanned.pages:
        legacy_bubbles: list[Bubble] = []
        for sb in sp.bubbles:
            lb = Bubble(
                idx=sb.idx,
                page_index=sb.page_index,
                polygon=sb.geometry.polygon,
                source_text=sb.source_text,
                ocr_confidence=sb.confidence,
            )
            bubble_map[id(lb)] = sb
            legacy_bubbles.append(lb)
        pages.append(Page(index=sp.index, bubbles=legacy_bubbles))
    return bubble_map, pages


def _build_result(
    scanned: ScannedChapter,
    pages: list[Page],
    bubble_map: dict[int, ScannedBubble],
) -> TranslatedChapter:
    translated_pages: list[TranslatedPage] = []
    for sp, lp in zip(scanned.pages, pages):
        translated_bubbles: list[TranslatedBubble] = []
        for lb in lp.bubbles:
            sb = bubble_map[id(lb)]
            translated_bubbles.append(TranslatedBubble(
                source=sb,
                translation_key=lb.translation_key or lb.id,
                translated_text=lb.translated_text or "",
                kind=lb.translation_status or "dialogue",
            ))
        translated_pages.append(TranslatedPage(
            source=sp,
            bubbles=tuple(translated_bubbles),
        ))
    return TranslatedChapter(scan=scanned, pages=tuple(translated_pages))


def _empty_result(scanned: ScannedChapter) -> TranslatedChapter:
    pages = tuple(
        TranslatedPage(source=sp, bubbles=())
        for sp in scanned.pages
    )
    return TranslatedChapter(scan=scanned, pages=pages)


def _page_windows(pages: list[Page]) -> list[list[Bubble]]:
    windows: list[list[Bubble]] = []
    current: list[Bubble] = []
    for page in pages:
        if current and len(current) + len(page.bubbles) > _PAGE_WINDOW_MAX_KEYS:
            windows.append(current)
            current = []
        current.extend(page.bubbles)
    if current:
        windows.append(current)
    return windows
