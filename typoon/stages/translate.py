"""Translate stage — scan.Chapter → (translate.Chapter, ChapterBrief).

Pure computation: no DB writes. Caller (worker) is responsible for persisting
the returned brief and translations.
"""

from __future__ import annotations

import asyncio
import unicodedata

from typoon.adapters.ctx import TranslateCtx
from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain import scan, translate
from typoon.domain.brief import ChapterBrief
from typoon.domain.scan import BubbleKey
from typoon.runs.artifacts import ArtifactSink
from typoon.stages.brief import build_chapter_brief
from typoon.stages.keys import assign_keys
from typoon.stages.page import TranslationOp, translate_window

# A window's source-text budget. The translator LLM has ~16k output token
# capacity; ~3000 source chars × ~1.5 expansion ratio + header overhead
# stays well below that with room for the brief slice on input.
_WINDOW_CHAR_BUDGET = 3000


async def translate_chapter(
    scanned: scan.Chapter,
    reader: PreparedReader,
    ctx: TranslateCtx,
    *,
    artifacts: ArtifactSink | None = None,
) -> tuple[translate.Chapter, ChapterBrief]:
    """Translate a scanned chapter. Returns (translated, brief).

    Does not persist. Caller is responsible for:

        await db.save_draft_brief(draft_id, brief.to_dict())
        await db.save_draft_bubbles(draft_id, translated.to_db_records())
    """
    if not scanned.all_bubbles:
        return _empty(scanned), ChapterBrief()

    keyed = assign_keys(scanned.all_bubbles, chapter_id=ctx.chapter_id)
    brief = await build_chapter_brief(ctx, reader, keyed, artifacts=artifacts)

    # Partition: noise bubbles get a free `kind="skip"` op (no LLM round
    # trip); the rest go through windowed translation.
    skip_keys = brief.noise_keys | {
        bk.key for bk in keyed if bk.bubble.page_index in brief.noise_pages
    }
    translatable = [bk for bk in keyed if bk.key not in skip_keys]
    ops: dict[str, TranslationOp] = {
        key: TranslationOp(key=key, kind="skip") for key in skip_keys
    }

    if not translatable:
        return _build(scanned, keyed, ops), brief

    windows = _make_windows(translatable)
    source_by_key = {bk.key: bk.source_text for bk in translatable}

    if artifacts is not None:
        _record_windows_plan(artifacts, keyed, translatable, skip_keys, windows)

    # Fan out windows in parallel. Provider errors propagate (chapter
    # fails); parse-incompleteness does not — translate_window returns
    # whatever it parsed and we collect missing keys for a single
    # combined retry below.
    batches = await asyncio.gather(*[
        translate_window(
            ctx, brief, window, translatable,
            window_num=i, total_windows=len(windows),
            artifacts=artifacts, window_tag=f"w{i:02d}",
        )
        for i, window in enumerate(windows)
    ])
    for batch in batches:
        for op in batch:
            ops[op.key] = op

    missing = [bk.key for bk in translatable if bk.key not in ops]
    if missing:
        if artifacts is not None:
            artifacts.write_json("06_translate", "missing_after_pass1.json", {
                "count":   len(missing),
                "keys":    missing,
                "sources": {k: source_by_key[k] for k in missing},
            })
        retry_ops = await translate_window(
            ctx, brief, missing, translatable,
            window_num=len(windows),
            total_windows=len(windows) + 1,
            artifacts=artifacts, window_tag="retry",
        )
        for op in retry_ops:
            ops[op.key] = op

        still_missing = [k for k in missing if k not in ops]
        if still_missing:
            if artifacts is not None:
                artifacts.write_json("06_translate", "unresolved.json", {
                    "count":   len(still_missing),
                    "keys":    still_missing,
                    "sources": {k: source_by_key[k] for k in still_missing},
                })
            # Every key here is real translatable content: noise was
            # filtered upstream by `brief.noise_keys` / `noise_pages`
            # before windowing. So "still missing" is unambiguously an
            # LLM or wire-format bug — prompt malformed, response
            # truncated by max_tokens, parser regex miss, provider
            # returned partial, model mirrored input header. The
            # artifacts under `06_translate/` (per-window
            # prompt/response/parsed, plus unresolved.json) are the
            # diagnosis surface.
            head = ", ".join(still_missing[:10])
            tail = f" (+{len(still_missing) - 10} more)" if len(still_missing) > 10 else ""
            raise RuntimeError(
                f"translate_chapter: {len(still_missing)} keys unresolved after retry: "
                f"{head}{tail}"
            )

    return _build(scanned, keyed, ops), brief


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(
    scanned: scan.Chapter,
    keyed: list[BubbleKey],
    ops: dict[str, TranslationOp],
) -> translate.Chapter:
    """Stitch per-key translation ops back onto the scanned bubble structure."""
    key_at = {(bk.page_index, bk.idx): bk.key for bk in keyed}
    pages = tuple(
        translate.Page(
            source=sp,
            bubbles=tuple(
                _materialize_bubble(sb, key_at, ops) for sb in sp.bubbles
            ),
        )
        for sp in scanned.pages
    )
    return translate.Chapter(scan=scanned, pages=pages)


def _materialize_bubble(
    sb: scan.Bubble,
    key_at: dict[tuple[int, int], str],
    ops: dict[str, TranslationOp],
) -> translate.Bubble:
    key = key_at.get((sb.page_index, sb.idx), f"p{sb.page_index}_b{sb.idx}")
    op = ops.get(key)
    text = op.text if op else ""
    return translate.Bubble(
        source=sb,
        translation_key=key,
        translated_text=_normalize_for_render(text),
        kind=op.kind if op else "skip",
    )


def _normalize_for_render(text: str) -> str:
    """NFC + typesetting post-process.

    Runs after LLM output, before render. Two jobs:
    1. NFC normalisation — force precomposed Vietnamese glyphs so the
       embedded render font can find them (LLM providers sometimes emit NFD).
    2. Typesetting cleanup — normalize punctuation to comic lettering
       standards so the render output looks professional regardless of
       what the LLM emitted.
    """
    if not text:
        return text
    import re
    # NFC first so subsequent regex operates on precomposed codepoints.
    text = unicodedata.normalize("NFC", text)

    # ── Ellipsis normalization ──────────────────────────────────────────
    # Collapse any run of 4+ dots (....., .............) to the two-beat
    # pause (……). Three dots stay as-is (standard hesitation ellipsis).
    # One or two dots are left alone (sentence-final or abbreviation).
    text = re.sub(r'\.{7,}', '……', text)   # 7+ dots → heavy pause
    text = re.sub(r'\.{4,6}', '…',  text)  # 4-6 dots → single ellipsis
    # Normalize Unicode ellipsis runs: …… is the max (two beats).
    text = re.sub(r'…{3,}', '……', text)

    # ── Trailing / leading whitespace per line ──────────────────────────
    text = '\n'.join(line.strip() for line in text.splitlines())

    # ── Strip blank lines ───────────────────────────────────────────────
    lines = [l for l in text.splitlines() if l]
    text = '\n'.join(lines)

    # ── Punctuation glued to last word, never stranded alone ───────────
    # Move a lone punctuation-only last line back onto the previous line.
    # e.g. "Không có gì\n." → "Không có gì."
    _PUNCT_ONLY = re.compile(r'^[.!?…,]+$')
    result_lines = lines[:]
    i = len(result_lines) - 1
    while i > 0 and _PUNCT_ONLY.match(result_lines[i]):
        result_lines[i - 1] = result_lines[i - 1] + result_lines[i]
        result_lines.pop(i)
        i -= 1
    text = '\n'.join(result_lines)

    return text


def _empty(scanned: scan.Chapter) -> translate.Chapter:
    return translate.Chapter(
        scan=scanned,
        pages=tuple(translate.Page(source=sp, bubbles=()) for sp in scanned.pages),
    )


def _make_windows(keyed: list[BubbleKey]) -> list[list[str]]:
    """Greedy-pack keys into windows whose source-char total stays under budget.

    Order is preserved: keys are sorted by `(page_index, idx)` so the
    translator sees bubbles in reading order, and each window is a
    contiguous slice of that order.
    """
    ordered = sorted(keyed, key=lambda bk: (bk.page_index, bk.idx))
    windows: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for bk in ordered:
        size = len(bk.source_text)
        if current and chars + size > _WINDOW_CHAR_BUDGET:
            windows.append(current)
            current, chars = [], 0
        current.append(bk.key)
        chars += size
    if current:
        windows.append(current)
    return windows


def _record_windows_plan(
    artifacts: ArtifactSink,
    keyed: list[BubbleKey],
    translatable: list[BubbleKey],
    skip_keys: set[str],
    windows: list[list[str]],
) -> None:
    chars_by_key = {bk.key: len(bk.source_text) for bk in translatable}
    artifacts.write_json("06_translate", "windows.json", {
        "total_windows": len(windows),
        "total_bubbles": len(keyed),
        "translatable":  len(translatable),
        "skip_keys":     sorted(skip_keys),
        "windows": [
            {
                "num":        i,
                "keys":       window,
                "char_count": sum(chars_by_key[k] for k in window),
            }
            for i, window in enumerate(windows)
        ],
    })
