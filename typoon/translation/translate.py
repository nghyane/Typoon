"""Chapter brief + keyed page translation pipeline."""

from __future__ import annotations

from typoon.app.events import PipelineError
from typoon.domain.bubble import Bubble, Page, Session

from .brief import ChapterBrief
from .context import build_chapter_brief
from .keys import assign_keys
from .page import translate_window

_PAGE_WINDOW_MAX_KEYS = 25


async def translate_pages(pages: list[Page], session: Session) -> tuple[int, Exception | None]:
    bubbles = [b for p in pages for b in p.bubbles]
    if not bubbles:
        return 0, None

    key_map = assign_keys(bubbles, project_id=session.project_id, chapter=session.chapter)
    turns = 0
    try:
        brief, used = await build_chapter_brief(pages, session, key_map)
        turns += used

        for window in _page_windows(pages):
            accepted = await translate_window(
                session, brief=brief, bubbles=window, key_map=key_map,
            )
            turns += 1
            for op in accepted:
                b = key_map[op.key]
                b.translation_status = op.status
                b.translated_text = op.text if op.status == "ok" else ""

        await session.store.save_chapter_brief(session.project_id, session.chapter, brief.to_dict())
        return turns, None
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        return turns, e


def _page_windows(pages: list[Page]) -> list[list[Bubble]]:
    windows: list[list[Bubble]] = []
    current: list[Bubble] = []
    for page in pages:
        if current and len(current) + len(page.bubbles) > _PAGE_WINDOW_MAX_KEYS:
            windows.append(current)
            current = []
        current.extend(page.bubbles)
        if len(current) >= _PAGE_WINDOW_MAX_KEYS:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows
