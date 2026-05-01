"""Translate stage — ScannedChapter → TranslatedChapter."""

from __future__ import annotations

import asyncio

from typoon.adapters.session import Session
from typoon.domain.scan import Bubble as ScannedBubble, Chapter as ScannedChapter
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter, Page as TranslatedPage
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import PipelineError
from typoon.agents import ChapterBrief, assign_keys, TranslationOp, translate_window, build_chapter_brief

_WINDOW_CHAR_BUDGET = 300  # max source chars of active keys per window


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

        windows = list(_windows(key_map, scanned))
        total = len(windows)
        
        # Run windows with limited concurrency — Cloudflare free tier rate limits
        _TRANSLATE_SEM = asyncio.Semaphore(4)
        
        async def _run_window(i: int, window_keys: list[str]) -> tuple[list[TranslationOp], int]:
            async with _TRANSLATE_SEM:
                return await translate_window(
                    session, brief=brief, window_keys=window_keys, key_map=key_map,
                    window_num=i, total_windows=total,
                )
        
        results = await asyncio.gather(*[
            _run_window(i, window_keys)
            for i, window_keys in enumerate(windows)
        ])
        
        ops: dict[str, TranslationOp] = {}
        for accepted, _ in results:
            for op in accepted:
                ops[op.key] = op

        await session.store.save_chapter_brief(
            session.project_id, session.chapter, brief.to_dict()
        )
    except Exception as e:
        session.hook.on(PipelineError(stage="translate", error=e))
        raise

    translated = _build(scanned, key_map, ops)

    from typoon.storage.records import translation_records
    records = translation_records(session.project_id, session.chapter, translated)
    await session.store.save_translations(session.project_id, session.chapter, records)

    return translated


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
    """Group keys into windows bounded by source char budget."""
    ordered = sorted(key_map.items(), key=lambda kv: (kv[1].page_index, kv[1].idx))

    windows: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for key, b in ordered:
        key_chars = len(b.source_text)
        if current and current_chars + key_chars > _WINDOW_CHAR_BUDGET:
            windows.append(current)
            current = []
            current_chars = 0
        current.append(key)
        current_chars += key_chars
    if current:
        windows.append(current)
    return windows
