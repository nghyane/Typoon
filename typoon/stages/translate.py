"""Translate stage — ScannedChapter → TranslatedChapter."""

from __future__ import annotations

import asyncio

from typoon.adapters.ctx import TranslateCtx
from typoon.stages.context import build_chapter_brief
from typoon.stages.keys import assign_keys
from typoon.stages.page import TranslationOp, translate_window
from typoon.domain.scan import BubbleKey, Chapter as ScannedChapter
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter, Page as TranslatedPage
from typoon.runs.artifacts import ArtifactSink

_WINDOW_CHAR_BUDGET = 600   # raised from 300 — fewer LLM calls per chapter


async def translate_chapter(
    scanned: ScannedChapter,
    ctx: TranslateCtx,
    *,
    artifacts: ArtifactSink | None = None,
) -> TranslatedChapter:
    """Translate a ScannedChapter. Raises on failure."""
    if not scanned.all_bubbles:
        return _empty(scanned)

    keyed = assign_keys(
        scanned.all_bubbles,
        project_id=ctx.project_id,
        chapter_id=ctx.chapter_id,
    )
    brief = await build_chapter_brief(ctx, scanned.prepared, keyed)

    windows = _make_windows(keyed)
    total   = len(windows)

    results = await asyncio.gather(*[
        translate_window(ctx, brief, wk, keyed, window_num=i, total_windows=total)
        for i, wk in enumerate(windows)
    ])

    ops: dict[str, TranslationOp] = {op.key: op for batch in results for op in batch}

    # Persist — brief first, then translations
    await ctx.store.save_chapter_brief(ctx.chapter_id, brief.to_dict())
    translated = _build(scanned, keyed, ops)
    await ctx.store.save_translations(ctx.chapter_id, translated.to_db_records())

    return translated


def _build(
    scanned: ScannedChapter,
    keyed: list[BubbleKey],
    ops: dict[str, TranslationOp],
) -> TranslatedChapter:
    pos_to_key = {(bk.page_index, bk.idx): bk.key for bk in keyed}
    pages = []
    for sp in scanned.pages:
        bubbles = []
        for sb in sp.bubbles:
            key = pos_to_key.get((sb.page_index, sb.idx), f"p{sb.page_index}_b{sb.idx}")
            op  = ops.get(key)
            bubbles.append(TranslatedBubble(
                source=sb,
                translation_key=key,
                translated_text=op.text if op else "",
                kind=op.kind if op else "skip",
            ))
        pages.append(TranslatedPage(source=sp, bubbles=tuple(bubbles)))
    return TranslatedChapter(scan=scanned, pages=tuple(pages))


def _empty(scanned: ScannedChapter) -> TranslatedChapter:
    return TranslatedChapter(
        scan=scanned,
        pages=tuple(TranslatedPage(source=sp, bubbles=()) for sp in scanned.pages),
    )


def _make_windows(keyed: list[BubbleKey]) -> list[list[str]]:
    ordered = sorted(keyed, key=lambda bk: (bk.page_index, bk.idx))
    windows: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for bk in ordered:
        n = len(bk.source_text)
        if current and chars + n > _WINDOW_CHAR_BUDGET:
            windows.append(current)
            current, chars = [], 0
        current.append(bk.key)
        chars += n
    if current:
        windows.append(current)
    return windows
