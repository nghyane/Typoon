"""Probe: test prompt variants on real bubbles to measure compound-title
pleonasm and long-sentence behavior.

Usage:
    python scripts/probe_prompt.py

Picks chapter 36 (work 11 / Chương 6, en→vi), runs 4 prompt variants
on the same target bubbles, prints side-by-side output. No DB writes,
no artifact pollution — pure read-and-compare.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from textwrap import shorten

from dotenv import load_dotenv

load_dotenv()

from typoon.config import load_config
from typoon.llm.ir import Message
from typoon.providers import make_translation_provider
from typoon.stages import prompt as prompt_mod

CHAPTER_ID = 36

# Bubbles known to expose the compound-title bug (page 0, indices 1/2/4).
TARGET_KEYS = ("p0_b1", "p0_b2", "p0_b4")


@dataclass(frozen=True)
class Variant:
    name:        str
    description: str
    extra_rules: str = ""   # appended after the source/target policy
    glossary:    dict[str, str] | None = None


VARIANTS: list[Variant] = [
    Variant(
        name="V0",
        description="baseline (current prompt, no changes)",
    ),
    Variant(
        name="V1",
        description="+ compound-title rule",
        extra_rules=(
            "## Compound titles\n\n"
            "English royal titles like 'His Majesty the King', 'Her Royal Highness Princess X', "
            "'His Imperial Majesty Emperor Y', 'His Royal Highness Crown Prince Y' are SINGLE "
            "titles — not noun phrases of two separate dignitaries. Translate to ONE Vietnamese "
            "title (`bệ hạ`, `điện hạ`, `hoàng thượng`, `thái tử`, ...). NEVER concatenate two "
            "Hán-Việt titles for one source compound (no `bệ hạ hoàng thượng`, no "
            "`điện hạ thái tử`)."
        ),
    ),
    Variant(
        name="V2",
        description="V1 + sentence-length rule",
        extra_rules=(
            "## Compound titles\n\n"
            "English royal titles like 'His Majesty the King', 'Her Royal Highness Princess X', "
            "'His Imperial Majesty Emperor Y', 'His Royal Highness Crown Prince Y' are SINGLE "
            "titles — not noun phrases of two separate dignitaries. Translate to ONE Vietnamese "
            "title (`bệ hạ`, `điện hạ`, `hoàng thượng`, `thái tử`, ...). NEVER concatenate two "
            "Hán-Việt titles for one source compound (no `bệ hạ hoàng thượng`, no "
            "`điện hạ thái tử`).\n\n"
            "## Bubble length\n\n"
            "Manhwa speech bubbles must be readable in a glance. If your Vietnamese translation "
            "of a single bubble exceeds ~20 words in one run-on sentence, split into 2–3 "
            "shorter sentences using `.` or `—`. The source may chain clauses with commas; "
            "Vietnamese should not mirror that."
        ),
    ),
    Variant(
        name="V3",
        description="V2 + glossary injection",
        extra_rules=(
            "## Compound titles\n\n"
            "English royal titles like 'His Majesty the King', 'Her Royal Highness Princess X', "
            "'His Imperial Majesty Emperor Y', 'His Royal Highness Crown Prince Y' are SINGLE "
            "titles — not noun phrases of two separate dignitaries. Translate to ONE Vietnamese "
            "title (`bệ hạ`, `điện hạ`, `hoàng thượng`, `thái tử`, ...). NEVER concatenate two "
            "Hán-Việt titles for one source compound (no `bệ hạ hoàng thượng`, no "
            "`điện hạ thái tử`).\n\n"
            "## Bubble length\n\n"
            "Manhwa speech bubbles must be readable in a glance. If your Vietnamese translation "
            "of a single bubble exceeds ~20 words in one run-on sentence, split into 2–3 "
            "shorter sentences using `.` or `—`. The source may chain clauses with commas; "
            "Vietnamese should not mirror that."
        ),
        glossary={
            "His Majesty the King":      "bệ hạ",
            "Her Royal Highness":        "điện hạ",
            "Her Highness":              "điện hạ",
            "Your Majesty":              "bệ hạ",
        },
    ),
]


async def main() -> None:
    import asyncpg

    config, _ = load_config()
    provider = make_translation_provider(config)
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    try:
        rows = await conn.fetch(
            "SELECT page_index, bubble_idx, source_text FROM bubbles "
            "WHERE chapter_id=$1 AND page_index=0 AND bubble_idx IN (0,1,2,3,4) "
            "ORDER BY bubble_idx",
            CHAPTER_ID,
        )
    finally:
        await conn.close()

    bubbles = [(f"p{r['page_index']}_b{r['bubble_idx']}", r["source_text"]) for r in rows]
    print(f"\n=== {len(bubbles)} bubbles loaded from chapter {CHAPTER_ID} ===\n")
    for k, t in bubbles:
        print(f"  [{k}] {shorten(t, 80)}")
    print()

    results: dict[str, dict[str, str]] = {k: {} for k, _ in bubbles}

    for variant in VARIANTS:
        print(f"\n--- running {variant.name}: {variant.description} ---")
        outputs = await run_variant(provider, variant, bubbles, "en", "vi")
        for key, text in outputs.items():
            results[key][variant.name] = text

    print_comparison(bubbles, results)


async def run_variant(
    provider,
    variant: Variant,
    bubbles: list[tuple[str, str]],
    source_lang: str,
    target_lang: str,
) -> dict[str, str]:
    """Send one prompt-variant call covering all bubbles. Returns key→translation."""
    system = build_system(variant, source_lang, target_lang)
    user = build_user(variant, bubbles)

    resp = await provider.call(
        [Message.system(system), Message.user_text(user)], [],
    )
    return parse_reply(resp.text or "", {k for k, _ in bubbles})


def build_system(variant: Variant, source_lang: str, target_lang: str) -> str:
    base = prompt_mod.PAGE_SYSTEM.format(
        source_lang_name=prompt_mod.lang_name(source_lang),
        target_lang_name=prompt_mod.lang_name(target_lang),
        source_policy=prompt_mod.load_source_policy(source_lang),
        target_policy=prompt_mod.load_target_policy(target_lang),
    )
    if variant.extra_rules:
        # Append the candidate rules at the end, before the final check
        # block. The final check is at the bottom; we insert before it.
        marker = "## Final check before replying"
        if marker in base:
            head, _, tail = base.partition(marker)
            return f"{head}{variant.extra_rules}\n\n{marker}{tail}"
        return f"{base}\n\n{variant.extra_rules}"
    return base


def build_user(variant: Variant, bubbles: list[tuple[str, str]]) -> str:
    """Wire bubbles into the input format the translator expects."""
    glossary_block = ""
    if variant.glossary:
        glossary_block = (
            "Glossary (use these mappings exactly when the source phrase appears):\n"
            + "\n".join(f"- {src} → {tgt}" for src, tgt in variant.glossary.items())
            + "\n\n"
        )

    lines = [glossary_block.strip(), ""] if glossary_block else []
    for key, text in bubbles:
        # Use a 7-char uppercase key shape so it matches the translator's
        # header regex. Convert p0_b1 → P00B0001 etc.
        wire_key = _to_wire_key(key)
        lines.append(f">>> {wire_key} page=0 active\n{text}")

    return "\n".join(lines)


def _to_wire_key(key: str) -> str:
    """`p0_b1` → 7-char [A-Z0-9] code matching the header regex.

    Format: P<page:2>B<idx:3> padded — enough variety for the probe.
    """
    p, b = key.split("_")
    p = int(p[1:])
    b = int(b[1:])
    return f"P{p:02d}B{b:03d}"


def parse_reply(text: str, valid_keys: set[str]) -> dict[str, str]:
    """Parse `@@ KEY kind\\nbody` blocks back into key→translation."""
    import re
    header_re = re.compile(r"^@@ ([A-Z0-9]{7}) (dialogue|sfx)\s*$", re.IGNORECASE)

    # Inverse mapping: wire key → original p0_b1 style
    rev = {_to_wire_key(k): k for k in valid_keys}

    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]

    out: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            if current is not None:
                wire = current
                if wire in rev:
                    out[rev[wire]] = "\n".join(body).strip()
            current = m.group(1)
            body = []
        elif current is not None:
            body.append(line)
    if current is not None and current in rev:
        out[rev[current]] = "\n".join(body).strip()
    return out


def print_comparison(
    bubbles: list[tuple[str, str]],
    results: dict[str, dict[str, str]],
) -> None:
    print("\n" + "=" * 90)
    print("COMPARISON")
    print("=" * 90)
    for key, src in bubbles:
        print(f"\n### {key}")
        print(f"SRC: {src}")
        for variant in VARIANTS:
            output = results[key].get(variant.name, "(missing)")
            print(f"\n  {variant.name} ({variant.description}):")
            for line in output.splitlines() or ["(empty)"]:
                print(f"    {line}")

        # Quick verdict per bubble
        print()
        verdict = analyze(key, src, results[key])
        for line in verdict:
            print(f"    {line}")


def analyze(key: str, src: str, outputs: dict[str, str]) -> list[str]:
    """Cheap heuristics to flag the bugs we're testing for."""
    lines = []
    has_compound = "his majesty the king" in src.lower() or "her royal highness" in src.lower()

    for name, vi in outputs.items():
        flags = []
        # Pleonasm: "bệ hạ" + "hoàng thượng" adjacent (within 5 chars)
        lower = vi.lower()
        for a, b in [("bệ hạ", "hoàng thượng"), ("điện hạ", "thái tử"),
                     ("hoàng thượng", "bệ hạ"), ("thái tử", "điện hạ")]:
            ia, ib = lower.find(a), lower.find(b)
            if ia >= 0 and ib >= 0 and abs(ia - ib) < 20:
                flags.append(f"pleonasm[{a}+{b}]")

        # Word count of longest sentence
        sentences = [s for s in vi.replace("—", ".").split(".") if s.strip()]
        max_words = max((len(s.split()) for s in sentences), default=0)
        if max_words >= 25:
            flags.append(f"long_sentence[{max_words}w]")

        flags_str = ", ".join(flags) if flags else "clean"
        lines.append(f"  {name}: {flags_str}")
    return lines


if __name__ == "__main__":
    asyncio.run(main())
