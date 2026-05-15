"""Audit existing translations with an LLM judge.

Phase 1 of the translation-quality investigation. No prompt changes, no
re-translation. Pure read-and-judge: for each sampled bubble, send
`(source_text, current_vi_text)` to a pro-translator judge agent and
record what issues it flags.

Output: `debug-runs/audit_<timestamp>/`
  - issues.jsonl       — one record per judged bubble
  - summary.json       — issue-type frequencies, per-chapter rates
  - prompt.txt         — exact judge prompt used (for reproducibility)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from typoon.config import load_config
from typoon.llm.ir import Message
from typoon.providers import _build_provider

# Sample N bubbles per chapter. 15 × ~28 chapters ≈ 420 judge calls.
DEFAULT_SAMPLE_PER_CHAPTER = 15

# Use full-size model for judge (Packyapi/gpt-5.4, not -mini).
JUDGE_MODEL = "Packyapi/gpt-5.4"

JUDGE_SYSTEM = """\
You are a professional Vietnamese comic / manhwa / manga translator and
editor. You have shipped many published volumes. You read scanlation
sources in English, Polish, Spanish, Chinese, Indonesian and other
languages and translate them to natural Vietnamese for an audience that
reads manhwa daily.

You are NOT translating. You are REVIEWING an already-published
translation pair: the OCR source on the left, the Vietnamese rendering
on the right. Decide whether the Vietnamese reads like work from a
competent fellow translator, or whether it has specific issues a real
editor would mark up.

For each pair, return a strict JSON object with this shape:

{
  "verdict":    "clean" | "minor" | "major",
  "issues":    [{"type": "<one of TYPE_LIST>", "severity": 1 | 2 | 3,
                 "evidence": "<short quote from the VI text>",
                 "fix": "<one-line suggested rewrite or rule>"}],
  "notes":     "<optional one-sentence overall comment>"
}

severity: 1 = nitpick, 2 = real issue, 3 = breaks readability.

TYPE_LIST (use exactly these strings — pick the closest fit):
  - "compound_title_pleonasm"
       Source has ONE compound dignitary title (e.g. "His Majesty the King",
       "Her Royal Highness Princess X") and VI translates it as TWO stacked
       Hán-Việt titles (e.g. "bệ hạ hoàng thượng", "điện hạ công chúa").
  - "literal_compound"
       Any other case where VI literally translates each source noun in a
       compound term, producing redundancy (e.g. "Mr. President" →
       "ngài tổng thống ông").
  - "honorific_doubled"
       Source honorific suffix (-san, -sama, -senpai, -님, -씨) AND the
       Vietnamese title for the same role both kept ("Ngài Tanaka-san").
  - "long_run_on"
       A single VI sentence over ~25 words mirroring source clause length
       instead of being split for readability.
  - "wrong_register"
       Pronoun / honorific level mismatches the scene (e.g. royal couple
       speaking with modern "em/anh", or strangers using "tao/mày").
  - "inconsistent_pronouns"
       Same speaker switches xưng hô mid-paragraph without reason.
  - "untranslated_or_kept_english"
       A source word that should have a natural VI rendering is left in
       the source language without good reason.
  - "machine_translation_flavour"
       Reads like word-for-word MT: rare collocations, awkward syntax,
       calques from source-language structure.
  - "missing_particles"
       VI is grammatically fine but lacks the conversational particles
       (à, nhé, thôi, mà, đấy, chứ, nhỉ) that make comic dialogue natural.
  - "wrong_kind"
       The translation treats dialogue as sfx or vice versa.
  - "other"
       Anything else worth flagging — explain in "fix".

If the translation is clean, return verdict "clean", empty issues array,
and no notes. Do NOT invent issues to fill space.

Reply with ONLY the JSON object. No markdown fences, no commentary.
"""

JUDGE_USER_TEMPLATE = """\
Source language: {source_lang}
Target language: Vietnamese
Bubble kind:     {kind}
Series context:  {series_label}

SOURCE:
{source}

VIETNAMESE:
{vi}
"""


async def main(sample_per_chapter: int, max_concurrent: int) -> None:
    import asyncpg

    config, _ = load_config()
    pcfg = config.providers[config.translation.provider]
    judge = _build_provider(
        config.translation.provider,
        pcfg,
        model=JUDGE_MODEL,
        reasoning_effort=None,
        max_tokens=4096,
    )

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        samples = await _collect_samples(conn, sample_per_chapter)
    finally:
        await conn.close()

    if not samples:
        print("no samples — exiting", file=sys.stderr)
        return

    run_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path("debug-runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(JUDGE_SYSTEM, "utf-8")

    print(f"[audit] {len(samples)} bubbles across "
          f"{len(set(s['chapter_id'] for s in samples))} chapters")
    print(f"[audit] judge model = {JUDGE_MODEL}")
    print(f"[audit] output → {out_dir}")
    print()

    sem = asyncio.Semaphore(max_concurrent)
    results: list[dict] = []
    t0 = time.monotonic()

    async def one(idx: int, sample: dict) -> None:
        async with sem:
            verdict = await _judge_one(judge, sample)
            results.append({**sample, **verdict})
            done = len(results)
            if done % 10 == 0 or done == len(samples):
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(samples) - done) / rate if rate else 0
                print(f"  [{done}/{len(samples)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    await asyncio.gather(*[one(i, s) for i, s in enumerate(samples)])

    # Write line-delimited JSON for easy diff/grep later
    with (out_dir / "issues.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarize(results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8",
    )

    _print_summary(summary)


async def _collect_samples(conn, k: int) -> list[dict]:
    """Pull k random translated bubbles per chapter that has a done draft."""
    chapters = await conn.fetch("""
        SELECT DISTINCT ON (c.id)
               c.id AS chapter_id, c.source_lang,
               wc.label, wc.work_id, td.id AS draft_id
        FROM translation_drafts td
        JOIN chapters c ON c.id = td.chapter_id
        JOIN work_chapters wc ON wc.id = c.work_chapter_id
        WHERE td.state = 'done'
        ORDER BY c.id, td.updated_at DESC
    """)
    samples: list[dict] = []
    for ch in chapters:
        if ch["source_lang"] is None:
            continue   # legacy, skip
        rows = await conn.fetch("""
            SELECT tdb.page_index, tdb.bubble_idx,
                   tdb.translated_text, tdb.kind,
                   b.source_text
            FROM translation_draft_bubbles tdb
            JOIN bubbles b
              ON b.chapter_id = $1
             AND b.page_index = tdb.page_index
             AND b.bubble_idx = tdb.bubble_idx
            WHERE tdb.draft_id = $2
              AND tdb.kind <> 'skip'
              AND tdb.translated_text <> ''
              AND length(b.source_text) >= 4
        """, ch["chapter_id"], ch["draft_id"])
        if not rows:
            continue
        picked = random.sample(list(rows), min(k, len(rows)))
        for r in picked:
            samples.append({
                "chapter_id":  ch["chapter_id"],
                "work_id":     ch["work_id"],
                "label":       ch["label"],
                "source_lang": ch["source_lang"],
                "page":        r["page_index"],
                "idx":         r["bubble_idx"],
                "kind":        r["kind"],
                "source":      r["source_text"],
                "vi":          r["translated_text"],
            })
    random.shuffle(samples)
    return samples


async def _judge_one(provider, sample: dict) -> dict:
    user = JUDGE_USER_TEMPLATE.format(
        source_lang=sample["source_lang"],
        kind=sample["kind"],
        series_label=sample["label"],
        source=sample["source"],
        vi=sample["vi"],
    )
    try:
        resp = await provider.call(
            [Message.system(JUDGE_SYSTEM), Message.user_text(user)], [],
        )
        text = (resp.text or "").strip()
        # Strip code fences if the judge ignored the instruction.
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        data = json.loads(text)
        return {
            "verdict": data.get("verdict", "unknown"),
            "issues":  data.get("issues", []),
            "notes":   data.get("notes", ""),
        }
    except Exception as e:
        return {"verdict": "error", "issues": [], "notes": f"judge error: {e}"}


def _summarize(results: list[dict]) -> dict[str, Any]:
    verdicts = Counter(r["verdict"] for r in results)
    issue_types = Counter()
    severity_by_type = defaultdict(list)
    by_lang_type: dict[str, Counter] = defaultdict(Counter)
    by_chapter: dict[int, dict] = defaultdict(
        lambda: {"label": "", "src": "", "n": 0, "issues": Counter()},
    )

    for r in results:
        ch = by_chapter[r["chapter_id"]]
        ch["label"] = r["label"]
        ch["src"]   = r["source_lang"]
        ch["n"]    += 1
        for it in r.get("issues", []):
            t = it.get("type", "other")
            sev = int(it.get("severity", 1))
            issue_types[t] += 1
            severity_by_type[t].append(sev)
            by_lang_type[r["source_lang"]][t] += 1
            ch["issues"][t] += 1

    return {
        "n_judged":       len(results),
        "verdicts":       dict(verdicts),
        "issue_types":    issue_types.most_common(),
        "avg_severity":   {
            t: round(sum(s) / len(s), 2)
            for t, s in severity_by_type.items()
        },
        "by_source_lang": {
            lang: dict(c.most_common()) for lang, c in by_lang_type.items()
        },
        "by_chapter": [
            {
                "chapter_id": cid,
                "label":      d["label"],
                "src":        d["src"],
                "n":          d["n"],
                "issues":     dict(d["issues"].most_common()),
            }
            for cid, d in sorted(by_chapter.items())
        ],
    }


def _print_summary(s: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"AUDIT SUMMARY  ({s['n_judged']} bubbles judged)")
    print("=" * 70)
    print("\nVerdicts:")
    for v, n in s["verdicts"].items():
        pct = n / s["n_judged"] * 100
        print(f"  {v:8s}  {n:4d}  ({pct:5.1f}%)")
    print("\nIssue types (most frequent first):")
    for t, n in s["issue_types"]:
        sev = s["avg_severity"].get(t, 0)
        print(f"  {t:30s}  {n:4d}   avg_sev={sev}")
    print("\nBy source language:")
    for lang, types in s["by_source_lang"].items():
        total = sum(types.values())
        print(f"  {lang}: {total} issues across {len(types)} types")
        for t, n in list(types.items())[:5]:
            print(f"      {t:28s}  {n}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-chapter", type=int, default=DEFAULT_SAMPLE_PER_CHAPTER)
    parser.add_argument("--concurrent",  type=int, default=6,
                        help="Max in-flight judge calls")
    args = parser.parse_args()

    random.seed(42)   # reproducible sampling
    asyncio.run(main(args.per_chapter, args.concurrent))
