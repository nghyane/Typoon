"""Probe both noise-handling approaches on real bubbles, side-by-side.

Bubble cases come from the audit results (chapters with embedded noise
or pure-chrome bubbles flagged wrong_kind / untranslated_or_kept_english).

We compare three variants on the same 30-ish bubbles:

  V0  baseline translator (current page.md)
  V1  translator + inline mask rule (handle embedded noise itself)
  V2  refine pass first, then translator

Output side-by-side: source -> V0 vi / V1 vi / V2 refined+vi.

Usage:
    python scripts/probe_noise_handling.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
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
from typoon.stages import prompt as prompt_mod

TRANSLATE_MODEL = "Packyapi/gpt-5.4-mini"
REFINE_MODEL    = "Packyapi/gpt-5.4-mini"

# Pull bubbles from these chapters because the audit flagged them
# with many wrong_kind / untranslated issues.
TARGET_CHAPTERS = [12, 13, 14, 11, 30]
BUBBLES_PER_CHAPTER = 6   # ~30 total

INLINE_MASK_RULE = """\

## Embedded noise inside dialogue bubbles

A speech bubble's OCR may contain non-diegetic chrome (watermarks,
scanlator credits, URLs, brand names, follow-us CTAs) glued onto the
real dialogue — typically prefix or suffix to the speech text. Examples:

  Raw: "TOTAL, NADIE ME VE. \\u3010菠萝包轻小说 BOOK.SFACG.COM"
  Real dialogue: "TOTAL, NADIE ME VE."

  Raw: "Cleaning: Cesar"
  Real dialogue: (none — entire bubble is chrome)

  Raw: "JOVEN MAESTRO POR FAVOR SIGA LA PAGINA DE FACEBOOK"
  Real dialogue: (none — entire bubble is a follow-us CTA)

When you see such mixed bubbles:
- Translate ONLY the in-story dialogue portion.
- Drop the chrome (watermarks, credits, URLs, brand names, follow-us CTAs).
- If the entire bubble is chrome, emit `@@ KEY skip` instead of dialogue/sfx.
  This is the ONLY case where you may emit `skip` — it must NOT be used
  for noisy-but-translatable text.

Use your judgment. If you cannot tell what is dialogue vs chrome,
prefer keeping the whole text (safer than dropping real speech).
"""

REFINE_SYSTEM = """\
You clean OCR text for a comic in {source_lang_name}. For each bubble,
output REFINED text:

1. Remove embedded non-diegetic chrome within a bubble: watermarks
   (`菠萝包轻小说`, `BOOK.SFACG.COM`), URLs (`www.baozimh.com`,
   `discord.gg/...`), scanlator brands (`BRS MANHUA`, `HEAVENLY DEMON
   SCANS`), credit lines (`Cleaning: Cesar`, `出品:快看漫画 作者:...`,
   `편집 지원 ...`), follow-us CTAs (`SIGA LA PAGINA DE FACEBOOK`,
   `JOIN US ON DISCORD`).

2. Fix obvious OCR errors: broken-word hyphens (`STRAŻ- NIK` ->
   `STRAŻNIK`), extra spaces inside CJK runs.

3. Do NOT translate. Do NOT change meaning. Do NOT add words.

4. If the entire bubble is non-diegetic chrome, emit empty body.

Conservative rule: if not 90% sure the change preserves meaning, keep raw.

Output: one block per input bubble, in input order:

```
@@ KEY
<refined text, or empty line for pure chrome>
```

Header is exactly `@@`, one space, the KEY copied verbatim. No commentary.
"""


def translator_system(source_lang: str, target_lang: str, with_inline_rule: bool) -> str:
    base = prompt_mod.PAGE_SYSTEM.format(
        source_lang_name=prompt_mod.lang_name(source_lang),
        target_lang_name=prompt_mod.lang_name(target_lang),
        source_policy=prompt_mod.load_source_policy(source_lang),
        target_policy=prompt_mod.load_target_policy(target_lang),
    )
    if with_inline_rule:
        # Insert before the final-check section so the final check still runs.
        marker = "## Final check before replying"
        if marker in base:
            head, _, tail = base.partition(marker)
            return f"{head}{INLINE_MASK_RULE}\n\n{marker}{tail}"
        return base + INLINE_MASK_RULE
    return base


async def main() -> int:
    import asyncpg
    config, _ = load_config()
    pcfg = config.providers[config.translation.provider]
    translator = _build_provider(
        config.translation.provider, pcfg,
        model=TRANSLATE_MODEL, reasoning_effort=None, max_tokens=8192,
    )
    refiner = _build_provider(
        config.translation.provider, pcfg,
        model=REFINE_MODEL, reasoning_effort=None, max_tokens=8192,
    )

    random.seed(42)
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        items = await _sample_bubbles(conn, TARGET_CHAPTERS, BUBBLES_PER_CHAPTER)
    finally:
        await conn.close()

    print(f"\n=== {len(items)} bubbles from chapters {TARGET_CHAPTERS} ===\n")

    out_dir = Path("debug-runs") / "probe_noise_handling"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print("[V0] baseline translate (no extra rule)...")
    v0 = await call_translate(translator, items, with_inline_rule=False)
    print(f"     done in {time.monotonic()-t0:.1f}s")

    t0 = time.monotonic()
    print("[V1] translate with inline-mask rule...")
    v1 = await call_translate(translator, items, with_inline_rule=True)
    print(f"     done in {time.monotonic()-t0:.1f}s")

    t0 = time.monotonic()
    print("[V2] refine then translate...")
    refined = await call_refine(refiner, items)
    refined_items = []
    for it in items:
        ref = refined.get(it["key"])
        refined_items.append({**it, "raw": ref if ref is not None else "",
                              "drop": ref is None})
    # Translate refined text — exclude pure-drop bubbles
    to_translate = [it for it in refined_items if not it["drop"] and it["raw"].strip()]
    v2_translated = await call_translate(translator, to_translate, with_inline_rule=False)
    v2 = {it["key"]: ("__DROP__" if it["drop"] else v2_translated.get(it["key"], "(missing)"))
          for it in refined_items}
    print(f"     done in {time.monotonic()-t0:.1f}s")

    _emit_comparison(items, v0, v1, v2, refined, out_dir)
    return 0


async def _sample_bubbles(conn, chapter_ids: list[int], k: int) -> list[dict]:
    items: list[dict] = []
    for cid in chapter_ids:
        ch = await conn.fetchrow(
            "SELECT id, source_lang FROM chapters WHERE id=$1", cid,
        )
        if ch is None or ch["source_lang"] is None:
            continue
        rows = await conn.fetch(
            "SELECT page_index, bubble_idx, source_text FROM bubbles "
            "WHERE chapter_id=$1 AND length(source_text) >= 4",
            cid,
        )
        if not rows:
            continue
        picked = random.sample(list(rows), min(k, len(rows)))
        for r in picked:
            items.append({
                "key":         _wire_key(r["page_index"], r["bubble_idx"]),
                "chapter":     cid,
                "page":        r["page_index"],
                "idx":         r["bubble_idx"],
                "source_lang": ch["source_lang"],
                "raw":         r["source_text"],
            })
    random.shuffle(items)
    return items


def _wire_key(page: int, idx: int) -> str:
    return f"P{page:02d}B{idx:03d}"


async def call_translate(
    provider,
    items: list[dict],
    *,
    with_inline_rule: bool,
) -> dict[str, str]:
    """Group items by source_lang (translator needs lang in prompt) and call
    once per language group."""
    by_lang: dict[str, list[dict]] = {}
    for it in items:
        by_lang.setdefault(it["source_lang"], []).append(it)

    out: dict[str, str] = {}
    tasks = []
    for lang, group in by_lang.items():
        tasks.append(_translate_group(provider, group, lang, with_inline_rule))
    results = await asyncio.gather(*tasks)
    for r in results:
        out.update(r)
    return out


async def _translate_group(
    provider, group: list[dict], source_lang: str, with_inline_rule: bool,
) -> dict[str, str]:
    system = translator_system(source_lang, "vi", with_inline_rule)
    blocks = [f">>> {it['key']} page={it['page']} active\n{it['raw']}" for it in group]
    user = "\n".join(blocks)

    resp = await provider.call(
        [Message.system(system), Message.user_text(user)], [],
    )
    return _parse_translate_blocks(resp.text or "", {it["key"] for it in group})


_TRANS_HEADER = re.compile(r"^@@ ([A-Z0-9]{7}) (dialogue|sfx|skip)\s*$", re.IGNORECASE)


def _parse_translate_blocks(text: str, valid_keys: set[str]) -> dict[str, str]:
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        if "\n" in text: text = text.split("\n", 1)[1]
        if text.endswith("```"): text = text[:-3]

    out: dict[str, str] = {}
    current: str | None = None
    kind = ""
    body: list[str] = []
    for line in text.splitlines():
        m = _TRANS_HEADER.match(line)
        if m:
            if current is not None and current in valid_keys and current not in out:
                if kind.lower() == "skip":
                    out[current] = "__SKIP__"
                else:
                    out[current] = "\n".join(body).strip()
            current = m.group(1)
            kind = m.group(2)
            body = []
        elif current is not None:
            body.append(line)
    if current is not None and current in valid_keys and current not in out:
        if kind.lower() == "skip":
            out[current] = "__SKIP__"
        else:
            out[current] = "\n".join(body).strip()
    return out


async def call_refine(provider, items: list[dict]) -> dict[str, str | None]:
    by_lang: dict[str, list[dict]] = {}
    for it in items:
        by_lang.setdefault(it["source_lang"], []).append(it)

    out: dict[str, str | None] = {}
    for lang, group in by_lang.items():
        system = REFINE_SYSTEM.format(source_lang_name=prompt_mod.lang_name(lang))
        blocks = [f">>> {it['key']}\n{it['raw']}" for it in group]
        user = "\n".join(blocks)
        resp = await provider.call(
            [Message.system(system), Message.user_text(user)], [],
        )
        out.update(_parse_refine_blocks(resp.text or "", {it["key"] for it in group}))
    return out


_REFINE_HEADER = re.compile(r"^@@\s+([A-Z0-9]{7})\s*$")


def _parse_refine_blocks(text: str, valid_keys: set[str]) -> dict[str, str | None]:
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        if "\n" in text: text = text.split("\n", 1)[1]
        if text.endswith("```"): text = text[:-3]

    out: dict[str, str | None] = {}
    current: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = _REFINE_HEADER.match(line)
        if m:
            if current is not None and current in valid_keys and current not in out:
                refined = "\n".join(body).strip()
                out[current] = refined if refined else None
            current = m.group(1)
            body = []
        elif current is not None:
            body.append(line)
    if current is not None and current in valid_keys and current not in out:
        refined = "\n".join(body).strip()
        out[current] = refined if refined else None
    return out


def _emit_comparison(
    items: list[dict],
    v0: dict[str, str],
    v1: dict[str, str],
    v2: dict[str, str],
    refined: dict[str, str | None],
    out_dir: Path,
) -> None:
    out_lines: list[str] = []
    for it in items:
        k = it["key"]
        v0_out = v0.get(k, "(missing)")
        v1_out = v1.get(k, "(missing)")
        v2_out = v2.get(k, "(missing)")
        ref = refined.get(k, "__NOPARSE__")

        out_lines.append(f"### {k} [c{it['chapter']} p{it['page']} b{it['idx']} {it['source_lang']}]")
        out_lines.append(f"   SRC:  {it['raw']!r}")
        if ref is None:
            out_lines.append(f"   REF:  (drop)")
        elif ref == "__NOPARSE__":
            out_lines.append(f"   REF:  (parse miss)")
        else:
            out_lines.append(f"   REF:  {ref!r}")
        out_lines.append(f"   V0:   {v0_out!r}")
        out_lines.append(f"   V1:   {v1_out!r}")
        out_lines.append(f"   V2:   {v2_out!r}")
        out_lines.append("")

    text = "\n".join(out_lines)
    (out_dir / "comparison.txt").write_text(text, "utf-8")

    # Brief metrics
    n = len(items)
    v0_filled = sum(1 for k in (i["key"] for i in items) if v0.get(k) and v0[k] != "__SKIP__")
    v0_skip   = sum(1 for k in (i["key"] for i in items) if v0.get(k) == "__SKIP__")
    v1_filled = sum(1 for k in (i["key"] for i in items) if v1.get(k) and v1[k] != "__SKIP__")
    v1_skip   = sum(1 for k in (i["key"] for i in items) if v1.get(k) == "__SKIP__")
    v2_filled = sum(1 for k in (i["key"] for i in items) if v2.get(k) and v2[k] not in ("__SKIP__", "__DROP__"))
    v2_drop   = sum(1 for k in (i["key"] for i in items) if v2.get(k) == "__DROP__")
    ref_drop  = sum(1 for k in (i["key"] for i in items) if refined.get(k) is None)
    ref_kept  = sum(1 for k in (i["key"] for i in items) if isinstance(refined.get(k), str) and refined[k] != "")

    summary = (
        f"\n=== Counts (n={n}) ===\n"
        f"V0 baseline:           translated={v0_filled}  skip={v0_skip}\n"
        f"V1 inline-mask:        translated={v1_filled}  skip={v1_skip}\n"
        f"V2 refine+translate:   translated={v2_filled}  dropped_by_refine={v2_drop}  refine_kept={ref_kept} refine_dropped={ref_drop}\n"
    )
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n" + text, "utf-8")
    print(f"\nartifacts → {out_dir}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
