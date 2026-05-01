"""Translate stage — ScannedChapter → TranslatedChapter."""

from __future__ import annotations

from typoon.adapters.session import Session
from typoon.domain.scan import Bubble as ScannedBubble, Chapter as ScannedChapter
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter, Page as TranslatedPage
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import PipelineError
from typoon.translation.brief import ChapterBrief
from typoon.translation.context import build_chapter_brief
from typoon.translation.keys import assign_keys
from typoon.translation.page import TranslationOp, translate_window

_PAGE_WINDOW_MAX_KEYS = 25


async def translate_chapter(
    scanned: ScannedChapter,
    session: Session,
    *,
    artifacts: ArtifactSink | None = None,
) -> TranslatedChapter:
    """Translate a ScannedChapter. Raises on agent failure."""
    all_bubbles = scanned.all_bubbles
    if not all_bubbles:
        return _empty(scanned)

    key_map = assign_keys(
        all_bubbles, project_id=session.project_id, chapter=session.chapter
    )

    try:
        brief, _ = await build_chapter_brief(session, scanned.prepared, key_map)

        ops: dict[str, TranslationOp] = {}
        for window_keys in _windows(key_map, scanned):
            accepted, _ = await translate_window(
                session, brief=brief, window_keys=window_keys, key_map=key_map,
            )
            for op in accepted:
                ops[op.key] = op

        await session.store.save_chapter_brief(
            session.project_id, session.chapter, brief.to_dict()
        )
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        raise

    return _build(scanned, key_map, ops)


# ── Build result ──────────────────────────────────────────────────────


def _build(
    scanned: ScannedChapter,
    key_map: dict[str, ScannedBubble],
    ops: dict[str, TranslationOp],
) -> TranslatedChapter:
    # Reverse map: bubble → key
    bubble_key: dict[int, str] = {
        id(b): key for key, b in key_map.items()
    }
    translated_pages = []
    for sp in scanned.pages:
        tbs = []
        for sb in sp.bubbles:
            key = bubble_key.get(id(sb), f"p{sb.page_index}_b{sb.idx}")
            op = ops.get(key)
            tbs.append(TranslatedBubble(
                source=sb,
                translation_key=key,
                translated_text=op.text if op else "",
                kind=op.kind if op else "skip",
            ))
        translated_pages.append(TranslatedPage(source=sp, bubbles=tuple(tbs)))
    return TranslatedChapter(scan=scanned, pages=tuple(translated_pages))


def _empty(scanned: ScannedChapter) -> TranslatedChapter:
    pages = tuple(TranslatedPage(source=sp, bubbles=()) for sp in scanned.pages)
    return TranslatedChapter(scan=scanned, pages=pages)


def _windows(
    key_map: dict[str, ScannedBubble],
    scanned: ScannedChapter,
) -> list[list[str]]:
    """Group keys into page-bounded windows of max _PAGE_WINDOW_MAX_KEYS."""
    # Build page → [key] map preserving bubble order
    page_keys: dict[int, list[str]] = {}
    for key, b in sorted(key_map.items(), key=lambda kv: (kv[1].page_index, kv[1].idx)):
        page_keys.setdefault(b.page_index, []).append(key)

    windows: list[list[str]] = []
    current: list[str] = []
    for sp in scanned.pages:
        keys_on_page = page_keys.get(sp.index, [])
        if current and len(current) + len(keys_on_page) > _PAGE_WINDOW_MAX_KEYS:
            windows.append(current)
            current = []
        current.extend(keys_on_page)
    if current:
        windows.append(current)
    return windows
