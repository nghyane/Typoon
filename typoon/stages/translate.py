"""Translate stage — scan.Chapter → (translate.Chapter, ChapterBrief).

Pure computation: no DB writes. Caller (worker) is responsible for persisting
the returned brief and translations.
"""

from __future__ import annotations

import asyncio

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain import scan, translate
from typoon.domain.scan import BubbleKey
from typoon.runs.artifacts import ArtifactSink
from typoon.stages.brief import ChapterBrief
from typoon.stages.context import build_chapter_brief
from typoon.stages.keys import assign_keys
from typoon.stages.page import TranslationOp, translate_window

_WINDOW_CHAR_BUDGET = 600


async def translate_chapter(
    scanned: scan.Chapter,
    reader: PreparedReader,
    ctx: TranslateCtx,
    *,
    artifacts: ArtifactSink | None = None,
) -> tuple[translate.Chapter, ChapterBrief]:
    """Translate a ScannedChapter. Returns (translated, brief) — does not persist.

    Caller is responsible for:
        await db.save_draft_brief(draft_id, brief.to_dict())
        await db.save_draft_bubbles(draft_id, translated.to_db_records())
    """
    if not scanned.all_bubbles:
        return _empty(scanned), ChapterBrief()

    keyed  = assign_keys(scanned.all_bubbles, chapter_id=ctx.chapter_id)
    brief  = await build_chapter_brief(ctx, scanned.prepared, reader, keyed)

    # Bubbles flagged as noise by the context agent (site chrome, watermarks,
    # buttons, page counters) bypass the translator entirely — they get a
    # kind="skip" op without an LLM round trip. Pages flagged whole as
    # noise extend that to every bubble on those pages.
    page_noise_keys = {
        bk.key for bk in keyed if bk.bubble.page_index in brief.noise_pages
    }
    skip_keys = brief.noise_keys | page_noise_keys
    translatable = [bk for bk in keyed if bk.key not in skip_keys]
    noise_ops: dict[str, TranslationOp] = {
        bk.key: TranslationOp(key=bk.key, kind="skip")
        for bk in keyed if bk.key in skip_keys
    }

    if not translatable:
        return _build(scanned, keyed, noise_ops), brief

    windows = _make_windows(translatable)
    total   = len(windows)

    # Fan out windows in parallel. Provider errors propagate (chapter fails);
    # parse-incompleteness does not — translate_window returns whatever it
    # parsed and we collect missing keys for a single combined retry below.
    results = await asyncio.gather(*[
        translate_window(ctx, brief, wk, translatable, window_num=i, total_windows=total)
        for i, wk in enumerate(windows)
    ])

    ops: dict[str, TranslationOp] = {**noise_ops}
    for batch in results:
        for op in batch:
            ops[op.key] = op

    missing = [bk.key for bk in translatable if bk.key not in ops]
    if missing:
        retry_ops = await translate_window(
            ctx, brief, missing, translatable,
            window_num=total, total_windows=total + 1,
        )
        for op in retry_ops:
            ops[op.key] = op
        still_missing = [k for k in missing if k not in ops]
        if still_missing:
            raise RuntimeError(
                f"translate_chapter: {len(still_missing)} keys unresolved after retry: "
                f"{', '.join(still_missing[:10])}"
                + (f" (+{len(still_missing) - 10} more)" if len(still_missing) > 10 else "")
            )

    return _build(scanned, keyed, ops), brief


def _build(
    scanned: scan.Chapter,
    keyed: list[BubbleKey],
    ops: dict[str, TranslationOp],
) -> translate.Chapter:
    pos_to_key = {(bk.page_index, bk.idx): bk.key for bk in keyed}
    pages = []
    for sp in scanned.pages:
        bubbles = []
        for sb in sp.bubbles:
            key = pos_to_key.get((sb.page_index, sb.idx), f"p{sb.page_index}_b{sb.idx}")
            op  = ops.get(key)
            bubbles.append(translate.Bubble(
                source=sb,
                translation_key=key,
                translated_text=op.text if op else "",
                kind=op.kind if op else "skip",
            ))
        pages.append(translate.Page(source=sp, bubbles=tuple(bubbles)))
    return translate.Chapter(scan=scanned, pages=tuple(pages))


def _empty(scanned: scan.Chapter) -> translate.Chapter:
    return translate.Chapter(
        scan=scanned,
        pages=tuple(translate.Page(source=sp, bubbles=()) for sp in scanned.pages),
    )


def _make_windows(keyed: list[BubbleKey]) -> list[list[str]]:
    ordered = sorted(keyed, key=lambda bk: (bk.page_index, bk.idx))
    windows: list[list[str]] = []
    current: list[str] = []
    chars   = 0
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
