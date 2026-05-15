"""Audit one chapter's translations with an LLM judge.

Designed to run per-chapter so many chapters can audit in parallel
without sharing state. Output is per-chapter: each judge result
streams to disk immediately so a killed process loses at most one
in-flight call.

Output (one directory per chapter):
    debug-runs/audit/<chapter_id>/
      bubbles.jsonl    — input (source, vi) pairs, one per line
      issues.jsonl     — judge verdicts, appended as each one finishes
      summary.json     — chapter-level rollup (written when all done)
      prompt.txt       — exact judge prompt used

Usage:
    python scripts/audit_chapter.py <chapter_id> [--sample N] [--concurrent N]
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
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from typoon.config import load_config
from typoon.llm.ir import Message
from typoon.providers import _build_provider


# Use full-size model for judge — quality of the audit matters more
# than speed since it runs offline.
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
  "verdict": "clean" | "minor" | "major",
  "issues": [{"type": "<one of TYPE_LIST>", "severity": 1 | 2 | 3,
              "evidence": "<short quote from the VI text>",
              "fix": "<one-line suggested rewrite or rule>"}],
  "notes": "<optional one-sentence overall comment>"
}

severity: 1 = nitpick, 2 = real issue, 3 = breaks readability.

TYPE_LIST (use exactly these strings — pick the closest fit):
  - "compound_title_pleonasm"
       Source has ONE compound dignitary title (e.g. "His Majesty the King")
       and VI translates it as TWO stacked Hán-Việt titles (e.g.
       "bệ hạ hoàng thượng").
  - "literal_compound"
       Any other case where VI literally translates each source noun in a
       compound term, producing redundancy.
  - "honorific_doubled"
       Source honorific suffix (-san, -sama, -senpai, -님, -씨) AND the
       Vietnamese title for the same role both kept.
  - "long_run_on"
       A single VI sentence over ~25 words mirroring source clause length
       instead of being split for readability.
  - "wrong_register"
       Pronoun / honorific level mismatches the scene.
  - "inconsistent_pronouns"
       Same speaker switches xưng hô mid-paragraph without reason.
  - "untranslated_or_kept_english"
       A source word that should have a natural VI rendering is left in
       the source language without good reason. Also applies to brand
       names ("Manhua", "Manga", "Baozi", scanlator credits) treated as
       in-story text.
  - "machine_translation_flavour"
       Reads like word-for-word MT: rare collocations, awkward syntax,
       calques from source-language structure.
  - "missing_particles"
       VI is grammatically fine but lacks the conversational particles
       (à, nhé, thôi, mà, đấy, chứ, nhỉ) that make comic dialogue natural.
  - "wrong_kind"
       The translation treats dialogue as sfx or vice versa, or treats
       chrome/watermark as in-story dialogue.
  - "dropped_subject"
       Source has an explicit subject/object that VI silently dropped,
       making the line ambiguous.
  - "other"
       Anything else worth flagging — explain in "fix".

If the translation is clean, return verdict "clean", empty issues array.
Do NOT invent issues to fill space.

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


async def main(chapter_id: int, sample: int | None, concurrent: int) -> int:
    import asyncpg

    config, _ = load_config()
    pcfg = config.providers[config.translation.provider]
    judge = _build_provider(
        config.translation.provider, pcfg,
        model=JUDGE_MODEL, reasoning_effort=None, max_tokens=4096,
    )

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        ch = await _load_chapter(conn, chapter_id)
        if ch is None:
            print(f"chapter {chapter_id}: no done draft — skipping", flush=True)
            return 2
        bubbles = await _load_bubbles(conn, ch["draft_id"], chapter_id, sample)
    finally:
        await conn.close()

    if not bubbles:
        print(f"chapter {chapter_id}: no translatable bubbles", flush=True)
        return 2

    out_dir = Path("debug-runs") / "audit" / str(chapter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(JUDGE_SYSTEM, "utf-8")

    # Persist input dataset for reproducibility.
    with (out_dir / "bubbles.jsonl").open("w", encoding="utf-8") as f:
        for b in bubbles:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    print(
        f"chapter {chapter_id} ({ch['source_lang']}): "
        f"{len(bubbles)} bubbles, concurrent={concurrent}, model={JUDGE_MODEL}",
        flush=True,
    )

    issues_path = out_dir / "issues.jsonl"
    # Append mode — if a previous run wrote partial results we keep them
    # only when explicitly continuing. For a clean run, overwrite first.
    issues_path.write_text("", "utf-8")

    sem  = asyncio.Semaphore(concurrent)
    lock = asyncio.Lock()
    results: list[dict] = []
    t0 = time.monotonic()

    async def judge_one(idx: int, bubble: dict) -> None:
        async with sem:
            verdict = await _judge(judge, bubble, ch)
        record = {**bubble, **verdict}
        async with lock:
            results.append(record)
            with issues_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done = len(results)
            if done % 5 == 0 or done == len(bubbles):
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(bubbles) - done) / rate if rate else 0
                print(
                    f"  [c{chapter_id}] {done}/{len(bubbles)} "
                    f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                    flush=True,
                )

    await asyncio.gather(*[judge_one(i, b) for i, b in enumerate(bubbles)])

    summary = _summarize(chapter_id, ch, results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8",
    )

    elapsed = time.monotonic() - t0
    print(
        f"chapter {chapter_id} done in {elapsed:.0f}s — "
        f"verdicts: {summary['verdicts']}",
        flush=True,
    )
    return 0


async def _load_chapter(conn, chapter_id: int) -> dict | None:
    row = await conn.fetchrow("""
        SELECT c.id AS chapter_id, c.source_lang,
               wc.label, wc.work_id, td.id AS draft_id
        FROM translation_drafts td
        JOIN chapters c ON c.id = td.chapter_id
        JOIN work_chapters wc ON wc.id = c.work_chapter_id
        WHERE c.id = $1 AND td.state = 'done'
        ORDER BY td.updated_at DESC LIMIT 1
    """, chapter_id)
    return dict(row) if row else None


async def _load_bubbles(conn, draft_id: int, chapter_id: int,
                        sample: int | None) -> list[dict]:
    rows = await conn.fetch("""
        SELECT tdb.page_index, tdb.bubble_idx, tdb.translated_text, tdb.kind,
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
        ORDER BY tdb.page_index, tdb.bubble_idx
    """, chapter_id, draft_id)
    bubbles = [{
        "page":   r["page_index"],
        "idx":    r["bubble_idx"],
        "kind":   r["kind"],
        "source": r["source_text"],
        "vi":     r["translated_text"],
    } for r in rows]
    if sample and len(bubbles) > sample:
        bubbles = random.sample(bubbles, sample)
        bubbles.sort(key=lambda b: (b["page"], b["idx"]))
    return bubbles


async def _judge(provider, bubble: dict, ch: dict) -> dict:
    user = JUDGE_USER_TEMPLATE.format(
        source_lang=ch["source_lang"],
        kind=bubble["kind"],
        series_label=ch["label"],
        source=bubble["source"],
        vi=bubble["vi"],
    )
    try:
        resp = await provider.call(
            [Message.system(JUDGE_SYSTEM), Message.user_text(user)], [],
        )
        text = (resp.text or "").strip()
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


def _summarize(chapter_id: int, ch: dict, results: list[dict]) -> dict:
    verdicts = Counter(r["verdict"] for r in results)
    issue_types: Counter = Counter()
    severity_by_type: dict[str, list[int]] = defaultdict(list)
    for r in results:
        for it in r.get("issues", []):
            t = it.get("type", "other")
            issue_types[t] += 1
            severity_by_type[t].append(int(it.get("severity", 1)))
    return {
        "chapter_id":   chapter_id,
        "work_id":      ch["work_id"],
        "label":        ch["label"],
        "source_lang":  ch["source_lang"],
        "n_judged":     len(results),
        "verdicts":     dict(verdicts),
        "issue_types":  issue_types.most_common(),
        "avg_severity": {
            t: round(sum(s) / len(s), 2)
            for t, s in severity_by_type.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_id", type=int)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N bubbles (default: all)")
    parser.add_argument("--concurrent", type=int, default=6)
    args = parser.parse_args()

    random.seed(42)
    sys.exit(asyncio.run(main(args.chapter_id, args.sample, args.concurrent)))
