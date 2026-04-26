"""Chapter brief + keyed page translation pipeline."""

from __future__ import annotations

from collections import defaultdict

from typoon.app.events import PipelineError
from typoon.domain.bubble import Bubble, Page, Session

from .brief import ChapterBrief
from .context import build_chapter_brief
from .keys import assign_keys
from .look_at import look_at_page
from .page import translate_window

_PAGE_WINDOW_MAX_KEYS = 25
_MAX_REPAIR_TURNS = 2


async def translate_pages(pages: list[Page], session: Session) -> tuple[int, Exception | None]:
    bubbles = [b for p in pages for b in p.bubbles]
    if not bubbles:
        return 0, None

    key_map = assign_keys(bubbles, project_id=session.project_id, chapter=_chapter(session))
    turns = 0
    try:
        brief, used = await build_chapter_brief(pages, session)
        turns += used
        turns += await _resolve_look_requests(session, brief, key_map, turn_base=turns)

        for window in _page_windows(pages):
            used = await _translate_window_until_done(session, brief, window, key_map, turns)
            turns += used

        await session.store.save_chapter_brief(session.project_id, _chapter(session), brief.to_dict())
        return turns, None
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        return turns, e


async def _translate_window_until_done(
    session: Session,
    brief: ChapterBrief,
    window: list[Bubble],
    key_map: dict[str, Bubble],
    turn_base: int,
) -> int:
    pending = {b.translation_key or "" for b in window}
    feedback: dict[str, str] = {}
    turns = 0
    for _ in range(_MAX_REPAIR_TURNS + 1):
        active_bubbles = [b for b in window if (b.translation_key or "") in pending]
        if not active_bubbles:
            break
        turns += 1
        accepted, need_look, invalid = await translate_window(
            session,
            brief=brief,
            bubbles=active_bubbles,
            key_map=key_map,
            look_notes=brief.key_notes,
            feedback=feedback,
            turn=turn_base + turns,
        )
        for op in accepted:
            b = key_map[op.key]
            b.translation_status = op.status
            b.translated_text = op.text if op.status == "ok" else ""
            pending.discard(op.key)
        if need_look:
            grouped: dict[int, list[str]] = defaultdict(list)
            for op in need_look:
                grouped[key_map[op.key].page_index].append(op.key)
            for page_index, keys in grouped.items():
                notes = await look_at_page(
                    session,
                    page_index=page_index,
                    keys=keys,
                    query="Clarify speaker, tone, local order, and whether text is dialogue or SFX/noise.",
                    source_by_key={k: key_map[k].source_text for k in keys},
                    turn=turn_base + turns,
                )
                brief.key_notes.update(notes)
        feedback = {k: invalid.get(k, "missing") for k in pending}
    if pending:
        raise RuntimeError(f"Untranslated keys: {', '.join(sorted(pending))}")
    return turns


async def _resolve_look_requests(
    session: Session,
    brief: ChapterBrief,
    key_map: dict[str, Bubble],
    *,
    turn_base: int,
) -> int:
    by_page: dict[int, list] = defaultdict(list)
    for req in brief.look_requests:
        by_page[req.page_index].append(req)
    turn = turn_base
    calls = 0
    for page_index, reqs in by_page.items():
        keys = sorted({k for req in reqs for k in req.keys if k in key_map})
        if not keys:
            continue
        query = "\n".join(req.query for req in reqs if req.query)
        turn += 1
        calls += 1
        notes = await look_at_page(
            session,
            page_index=page_index,
            keys=keys,
            query=query or "Clarify visual context for marked keys.",
            source_by_key={k: key_map[k].source_text for k in keys},
            turn=turn,
        )
        brief.key_notes.update(notes)
    return calls


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


def _chapter(session: Session) -> float:
    return float(getattr(session, "chapter", 0.0))
