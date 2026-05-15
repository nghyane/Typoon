"""Refine PoC — clean OCR text before translate.

Hypothesis: many translation issues (wrong_kind, untranslated_or_kept_english,
machine_translation_flavour) come from raw OCR text containing embedded
noise (watermarks, credits, URLs ghép cùng thoại) or OCR errors (broken
words, mixed scripts). A small LLM pass that REFINES each bubble's text
— without classifying it — should clean both at once.

This PoC:
  1. Loads all bubbles for a chapter.
  2. Batches them into windows under a char budget.
  3. Sends each window to the model with a refine prompt.
  4. Parses back refined text per bubble.
  5. Diffs source vs refined, writes results to debug-runs/refine_poc_c<id>/.

No DB writes. No production hook. Read-only audit of what refine could do.

Usage:
    python scripts/refine_poc.py <chapter_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from textwrap import shorten

from dotenv import load_dotenv

load_dotenv()

from typoon.config import load_config
from typoon.llm.ir import Message
from typoon.providers import _build_provider
from typoon.stages.prompt import lang_name


# Reuse the translator's mini for refine — refine is mechanical
# (rewrite text, drop noise, fix typos), so the small model is enough.
REFINE_MODEL = "Packyapi/gpt-5.4-mini"

# Per-window source-char budget. Generous (refine reply ≈ same size as
# input, well under 16k output cap) — and refine has NO context block,
# unlike translate, so a window can hold more bubbles.
WINDOW_CHAR_BUDGET = 5000

REFINE_SYSTEM = """\
You are a comic OCR text cleaner. You receive raw OCR output for speech
bubbles from a {source_lang_name} comic page. For each bubble, output a
REFINED version of the text:

1. Remove embedded non-diegetic noise WITHIN a bubble: watermarks
   (`菠萝包轻小说`, `BOOK.SFACG.COM`, `BRS MANHUA`), site URLs
   (`www.baozimh.com`, `discord.gg/...`), scanlator brand names
   (`HEAVENLY DEMON SCANS`, `DRAGON COMICS AGE`), credit lines
   (`Cleaning: Cesar`, `Typesetting: Morfeusz`, `出品:快看漫画 作者:...`,
   `편집 지원 Redice Studio`), follow-us CTAs in any language
   (`SIGA LA PAGINA DE FACEBOOK`, `POTRZEBUJEMY OPINII`,
   `JOIN US ON DISCORD`).

2. Fix obvious OCR errors:
   - Broken words split by hyphen + space: `STRAŻ- NIK` -> `STRAŻNIK`
   - Stray characters from a different script that look like noise
     (Cyrillic `ТУКА` inside Polish): correct to the intended script
     when you can guess unambiguously; otherwise leave alone.
   - Extra spaces inside CJK runs.
   - Do NOT translate. Do NOT change meaning. Do NOT add words.

3. If the ENTIRE bubble is non-diegetic chrome (pure watermark, pure
   credit, pure title card, pure URL, pure scanlator brand), output
   an empty refined value — translator will skip it.

Conservative rule: if you are not at least 90% sure the change preserves
the original dialogue meaning, KEEP the raw text unchanged. False refines
cost more than no refine.

Output format (STRICT). For each input bubble, emit ONE block:

```
@@ KEY
<refined text, or empty line for pure noise>
```

Rules:
- Header is exactly `@@`, one space, the KEY copied verbatim.
- Body is the refined text. May span multiple lines.
- Empty body means "pure noise, drop this bubble".
- Every active input bubble MUST produce exactly one `@@ KEY` block.
- No commentary, no markdown fences, no `>>>` echoed back.
"""

REFINE_USER_HEADER = """\
Refine these {n} bubbles. Source language: {source_lang_name}. Reply with
one `@@ KEY` block per bubble in the same order.
"""

_HEADER_RE = re.compile(r"^@@\s+([A-Z0-9]{7})\s*$")


async def main(chapter_id: int) -> int:
    import asyncpg
    config, _ = load_config()
    pcfg = config.providers[config.translation.provider]
    provider = _build_provider(
        config.translation.provider, pcfg,
        model=REFINE_MODEL, reasoning_effort=None, max_tokens=8192,
    )

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        info = await conn.fetchrow(
            "SELECT id, source_lang, page_count FROM chapters WHERE id=$1",
            chapter_id,
        )
        if info is None:
            print(f"chapter {chapter_id} not found", file=sys.stderr)
            return 2
        bubbles = await conn.fetch(
            "SELECT page_index, bubble_idx, source_text "
            "FROM bubbles WHERE chapter_id=$1 ORDER BY page_index, bubble_idx",
            chapter_id,
        )
    finally:
        await conn.close()

    source_lang = info["source_lang"] or "en"
    items = [
        {
            "key":  _make_key(b["page_index"], b["bubble_idx"]),
            "page": b["page_index"],
            "idx":  b["bubble_idx"],
            "raw":  b["source_text"] or "",
        }
        for b in bubbles
    ]
    items = [it for it in items if it["raw"].strip()]
    print(f"chapter {chapter_id}: {len(items)} bubbles ({source_lang})")

    out_dir = Path("debug-runs") / f"refine_poc_c{chapter_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _make_windows(items, WINDOW_CHAR_BUDGET)
    print(f"split into {len(windows)} window(s) under {WINDOW_CHAR_BUDGET} chars")

    t0 = time.monotonic()
    refined: dict[str, str | None] = {}
    for i, win in enumerate(windows):
        win_t0 = time.monotonic()
        result = await refine_window(provider, win, source_lang, i, out_dir)
        refined.update(result)
        print(f"  window {i+1}/{len(windows)}: {len(win)} bubbles, "
              f"{len(result)} parsed ({(time.monotonic()-win_t0):.1f}s)")
    elapsed = time.monotonic() - t0

    # Build diff report
    report = _build_report(items, refined)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8",
    )
    (out_dir / "diff.txt").write_text(_format_diff(items, refined), "utf-8")

    _print_summary(report, elapsed)
    print(f"\nartifacts → {out_dir}")
    return 0


def _make_key(page: int, idx: int) -> str:
    """7-char key matching the wire format regex."""
    return f"P{page:02d}B{idx:03d}"


def _make_windows(
    items: list[dict],
    budget: int,
) -> list[list[dict]]:
    windows: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for it in items:
        n = len(it["raw"])
        if current and chars + n > budget:
            windows.append(current)
            current, chars = [], 0
        current.append(it)
        chars += n
    if current:
        windows.append(current)
    return windows


async def refine_window(
    provider,
    items: list[dict],
    source_lang: str,
    window_idx: int,
    out_dir: Path,
) -> dict[str, str | None]:
    """Send one window to the model, return key→refined_text (or None on parse miss)."""
    system = REFINE_SYSTEM.format(source_lang_name=lang_name(source_lang))
    user_lines = [REFINE_USER_HEADER.format(
        n=len(items), source_lang_name=lang_name(source_lang),
    )]
    for it in items:
        user_lines.append(f">>> {it['key']}\n{it['raw']}")
    user = "\n".join(user_lines)

    (out_dir / f"w{window_idx:02d}_prompt.txt").write_text(user, "utf-8")
    (out_dir / f"w{window_idx:02d}_system.txt").write_text(system, "utf-8")

    resp = await provider.call(
        [Message.system(system), Message.user_text(user)], [],
    )
    raw_reply = resp.text or ""
    (out_dir / f"w{window_idx:02d}_response.txt").write_text(raw_reply, "utf-8")

    return _parse_blocks(raw_reply, {it["key"] for it in items})


def _parse_blocks(text: str, valid_keys: set[str]) -> dict[str, str | None]:
    """Parse `@@ KEY\\nbody...` blocks. Empty body → None (drop)."""
    # Strip <think> + code fences
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]

    out: dict[str, str | None] = {}
    current: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            if current is not None:
                _commit(out, current, body, valid_keys)
            current = m.group(1)
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        _commit(out, current, body, valid_keys)
    return out


def _commit(
    out: dict[str, str | None],
    key: str,
    body: list[str],
    valid_keys: set[str],
) -> None:
    if key not in valid_keys or key in out:
        return
    refined = "\n".join(body).strip()
    out[key] = refined if refined else None


def _build_report(items: list[dict], refined: dict[str, str | None]) -> dict:
    """Categorize each bubble's outcome."""
    unchanged = []
    cleaned = []
    dropped = []
    missing = []
    grew = []

    for it in items:
        ref = refined.get(it["key"], "__MISSING__")
        if ref == "__MISSING__":
            missing.append(it)
            continue
        if ref is None:
            dropped.append(it)
            continue
        if ref == it["raw"].strip():
            unchanged.append(it)
            continue
        if len(ref) > len(it["raw"]) * 1.2:
            grew.append({**it, "refined": ref})
            continue
        cleaned.append({**it, "refined": ref})

    return {
        "n_total":     len(items),
        "unchanged":   len(unchanged),
        "cleaned":     len(cleaned),
        "dropped":     len(dropped),
        "grew":        len(grew),
        "missing":     len(missing),
        "cleaned_samples": [
            {"key": c["key"], "raw": c["raw"][:200], "refined": c["refined"][:200]}
            for c in cleaned[:30]
        ],
        "dropped_samples": [
            {"key": d["key"], "raw": d["raw"][:200]}
            for d in dropped[:30]
        ],
        "grew_samples": [
            {"key": g["key"], "raw": g["raw"][:200], "refined": g["refined"][:200]}
            for g in grew[:10]
        ],
        "missing_samples": [
            {"key": m["key"], "raw": m["raw"][:200]}
            for m in missing[:10]
        ],
    }


def _format_diff(items: list[dict], refined: dict[str, str | None]) -> str:
    lines = []
    for it in items:
        ref = refined.get(it["key"], "__MISSING__")
        if ref == "__MISSING__":
            tag = "MISSING"; new = ""
        elif ref is None:
            tag = "DROP   "; new = ""
        elif ref == it["raw"].strip():
            continue  # unchanged → skip in diff for readability
        else:
            tag = "CLEAN  "; new = ref

        lines.append(f"[{tag}] {it['key']}  p{it['page']} b{it['idx']}")
        lines.append(f"   raw:     {shorten(it['raw'], 180)!r}")
        if new:
            lines.append(f"   refined: {shorten(new, 180)!r}")
        lines.append("")
    return "\n".join(lines)


def _print_summary(report: dict, elapsed: float) -> None:
    print()
    print("=" * 60)
    print(f"REFINE PoC SUMMARY  (elapsed {elapsed:.1f}s)")
    print("=" * 60)
    n = report["n_total"]
    for key in ("unchanged", "cleaned", "dropped", "grew", "missing"):
        v = report[key]
        pct = v / n * 100 if n else 0
        print(f"  {key:10s}  {v:4d}  ({pct:5.1f}%)")

    print("\nSample cleaned (raw → refined):")
    for s in report["cleaned_samples"][:5]:
        print(f"  [{s['key']}]")
        print(f"    raw:     {shorten(s['raw'], 100)!r}")
        print(f"    refined: {shorten(s['refined'], 100)!r}")

    print("\nSample dropped (pure noise):")
    for s in report["dropped_samples"][:5]:
        print(f"  [{s['key']}] {shorten(s['raw'], 120)!r}")

    if report["grew_samples"]:
        print("\nSample grew (suspicious — may have hallucinated):")
        for s in report["grew_samples"][:5]:
            print(f"  [{s['key']}]")
            print(f"    raw:     {shorten(s['raw'], 100)!r}")
            print(f"    refined: {shorten(s['refined'], 100)!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_id", type=int)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.chapter_id)))
