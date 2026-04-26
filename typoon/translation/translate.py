"""Chapter brief + keyed page translation pipeline."""

from __future__ import annotations

from typoon.app.events import PipelineError
from typoon.domain.bubble import Page, Session

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
        brief, ctx_turns = await build_chapter_brief(pages, session, key_map)
        turns += ctx_turns

        for window in _page_windows(pages):
            accepted, page_turns = await translate_window(
                session, brief=brief, bubbles=window, key_map=key_map,
                all_pages=pages,
            )
            turns += page_turns
            for op in accepted:
                b = key_map[op.key]
                b.translation_status = op.status
                b.translated_text = op.text if op.status == "ok" else ""

        await session.store.save_chapter_brief(session.project_id, session.chapter, brief.to_dict())
        return turns, None
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        return turns, e


def _page_windows(pages: list[Page]) -> list[list]:
    """Group bubbles into windows, preferring page boundaries."""
    windows: list[list] = []
    current: list = []
    for page in pages:
        if current and len(current) + len(page.bubbles) > _PAGE_WINDOW_MAX_KEYS:
            windows.append(current)
            current = []
        current.extend(page.bubbles)
    if current:
        windows.append(current)
    return windows
