"""Probe: deterministic speaker inference from text + bubble adjacency.

Inputs: real scan output (bubble text, page, idx, polygon, shape_kind).
NO LLM, NO vision call.

Algorithm (idea 2 + 6 simplified):
  1. Caption / narration detection — wide rectangular bubbles with no
     pointing tail tend to be narration. We don't have tail data, but we
     can flag bubbles whose polygon aspect ratio is wide-and-flat AND
     positioned at page top/bottom edges (classic caption placement).
  2. Internal monologue — first-person past/future tense in dialogue
     bubbles. "I'd", "I'll", "I'm" → likely viewpoint character speaker.
  3. Vocative addressee — bubble text ending with "<NAME>!" or starting
     with "<NAME>," — addressee = NAME, signals listener for adjacent
     bubble.
  4. Alternation graph — sequential bubbles on same page alternate
     speakers when neither is narration. Anchor one bubble with a heuristic
     signal, propagate via alternation.
  5. Cross-bubble repetition — bubbles starting with identical first
     3 words within a page → same speaker continuation.

Output: same shape as combined probe (key → speaker). Compare with
combined probe's 34/37 to see how much deterministic alone covers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_probe_3x3 import WebpPreparedReader, OUT

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHAP = ROOT / "cache" / "probe_chapter"
SLICE = list(sorted(CHAP.glob("*.png")))[5:14]

# Combined-probe ground truth, hand-extracted from probe_combined.md for the
# same 9 pages. Used to score the deterministic-only approach.
COMBINED_GT: dict[str, str] = {
    "9TR2GDU": "Denji", "WZ5CENQ": "Denji", "K9MRRAN": "Denji", "ELCDRVS": "Denji",
    "VUC2XMK": "Denji", "KCCBHK2": "Denji", "3BGG85U": "Denji", "SCS7JUM": "Denji",
    "WJLA2BC": "Pochita", "7HSBF6K": "sfx",
    "5K3VUJA": "unknown", "GJUKQZS": "unknown", "N3YURXZ": "unknown",
    "RMNHFDH": "Denji", "RNTBPWA": "Denji",
    "9NCW8DT": "Pochita",
    "UUANZTS": "Denji", "8985Y6Y": "Denji",
    "H8DTPBW": "Pochita",
    "HKKVWQJ": "Denji", "FRW3HYD": "Pochita", "TYA4MMD": "Pochita",
    "S7L7N56": "Denji", "MFES64K": "Pochita",
    "QK3NCAR": "Denji", "6DE85L2": "Pochita", "8C8ZDGC": "Denji",
    "8RELWDA": "Denji", "EE67AGP": "Denji", "C4UW7VM": "Denji", "KL9GLFW": "Denji",
    "ZU6ZH2B": "Denji", "GWPGQBZ": "Denji", "T4M7PG4": "Denji",
    "ACC78YA": "Denji", "MGDT96M": "Denji", "54VCME6": "Pochita",
}


# ── Deterministic signals ───────────────────────────────────────────


_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|I'm|I'll|I'd|I've|me|my|mine|myself)\b",
    re.IGNORECASE,
)
_VOCATIVE_END_RE = re.compile(r"\b([A-Z][a-zA-Z]+)[!,.\s]*$")
_VOCATIVE_START_RE = re.compile(r"^([A-Z][a-zA-Z]+),")
# Scanlation/site chrome noise patterns — extends noise_terms.txt.
_NOISE_PATTERNS = [
    re.compile(r"do\s+not\s+mirror", re.IGNORECASE),
    re.compile(r"\.(com|net|org|tv)\b", re.IGNORECASE),
    re.compile(r"mangastream", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
]


def is_noise(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    return any(p.search(s) for p in _NOISE_PATTERNS)


def is_sfx_text(text: str) -> bool:
    """Heuristic SFX: very short, all-caps onomatopoeia or non-alphabetic."""
    s = text.strip()
    if not s or len(s) > 12:
        return False
    # All caps onomatopoeia like "WHIMPER", "WOOF", "BANG"
    if s.isupper() and s.replace(" ", "").isalpha():
        return True
    return False


def is_caption_shape(bubble: dict, page_w: int, page_h: int) -> bool:
    """Caption box heuristic: wide rectangle, no tail, near page edge.

    We don't have tail data, so we approximate with aspect ratio + position.
    Caption boxes are typically ≥3:1 wide and within ~15% of page edge.
    """
    poly = bubble["polygon"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h <= 0 or w / h < 2.5:
        return False
    # Near top/bottom edge?
    cy = (max(ys) + min(ys)) / 2
    top_edge = cy < page_h * 0.20
    bot_edge = cy > page_h * 0.80
    return top_edge or bot_edge


# ── Assignment ──────────────────────────────────────────────────────


def assign_deterministic(
    flat: list[dict],
    page_dims: list[tuple[int, int]],
    viewpoint: str = "protagonist",
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (speaker_per_key, reason_per_key)."""
    speakers: dict[str, str] = {}
    reason: dict[str, str] = {}

    # Pass 1: pure-text classification.
    for b in flat:
        text = b["text"]
        if is_noise(text):
            speakers[b["key"]] = "noise"
            reason[b["key"]] = "noise pattern (regex)"
            continue
        if is_sfx_text(text):
            speakers[b["key"]] = "sfx"
            reason[b["key"]] = "all-caps onomatopoeia"
            continue
        # Caption shape → narration.
        page_w, page_h = page_dims[b["page"]]
        if is_caption_shape(b, page_w, page_h):
            speakers[b["key"]] = "narrator"
            reason[b["key"]] = "wide rectangle at page edge"
            continue
        # First-person dialogue → viewpoint character.
        if _FIRST_PERSON_RE.search(text):
            speakers[b["key"]] = viewpoint
            reason[b["key"]] = "first-person marker"
            continue

    # Pass 2: alternation propagation within page.
    # Group by page, sorted by reading order (idx).
    by_page: dict[int, list[dict]] = {}
    for b in flat:
        by_page.setdefault(b["page"], []).append(b)

    for page, bubbles in by_page.items():
        bubbles.sort(key=lambda x: x.get("idx", 0))
        # Walk forward, when we hit a bubble with no speaker assigned but
        # the prior bubble has a non-narrator speaker, alternate it.
        prev_speaker: str | None = None
        for b in bubbles:
            cur = speakers.get(b["key"])
            if cur in (None, "unknown"):
                if prev_speaker and prev_speaker not in ("narrator", "sfx", "noise"):
                    # Alternate — but we don't know the "other" speaker.
                    # Mark as "other" placeholder; resolved later if we
                    # have a known two-character context.
                    speakers[b["key"]] = "other"
                    reason[b["key"]] = f"alternation from prev={prev_speaker}"
            cur = speakers.get(b["key"])
            if cur not in (None, "other", "narrator", "sfx", "noise"):
                prev_speaker = cur

    # Default unassigned to unknown.
    for b in flat:
        if b["key"] not in speakers:
            speakers[b["key"]] = "unknown"
            reason[b["key"]] = "no signal"

    return speakers, reason


# ── Driver ──────────────────────────────────────────────────────────


async def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]

    reader = WebpPreparedReader(SLICE)
    prepared = reader.chapter("chainsaw")
    log.info("scanning…")
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    scan_t = time.monotonic() - t0
    log.info("scan: %.1fs, %d bubbles", scan_t, len(out.chapter.all_bubbles))

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    flat: list[dict] = []
    for bk in keyed:
        b = bk.bubble
        flat.append({
            "key": bk.key,
            "page": b.page_index,
            "idx": b.idx,
            "text": b.source_text,
            "shape_kind": b.shape_kind,
            "polygon": b.box.polygon,
        })

    page_dims = [(p.width, p.height) for p in reader._pages]

    # For Chainsaw Man, viewpoint=Denji is correct. In a real pipeline this
    # would come from material memory after pre-analyze.
    t0 = time.monotonic()
    speakers, reason = assign_deterministic(flat, page_dims, viewpoint="Denji")
    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info("deterministic: %.2f ms", elapsed_ms)

    # Score vs combined-probe ground truth.
    correct = 0
    wrong = 0
    nomatch = 0
    for b in flat:
        gt = COMBINED_GT.get(b["key"])
        pred = speakers.get(b["key"], "unknown")
        # Normalize: noise/sfx treated as same family for fairness
        gt_norm = gt
        pred_norm = pred
        if gt is None:
            nomatch += 1
            continue
        if gt_norm == pred_norm:
            correct += 1
        elif gt_norm == "unknown" and pred_norm in ("unknown", "other"):
            correct += 1  # both said unknown
        elif gt_norm == "sfx" and pred_norm in ("sfx", "noise"):
            correct += 1
        else:
            wrong += 1

    log.info("score vs combined probe: correct=%d wrong=%d (no-gt=%d) of %d",
             correct, wrong, nomatch, len(flat))

    # Write report.
    report = [
        "# Deterministic speaker probe (no LLM, no vision)",
        "",
        f"- bubbles: {len(flat)}",
        f"- viewpoint (pre-seeded): Denji",
        f"- compute time: {elapsed_ms:.2f} ms",
        f"- correct vs combined GT: {correct}/{len(flat)} ({correct/len(flat)*100:.0f}%)",
        f"- wrong: {wrong}",
        "",
        "| key | page | text | GT | deterministic | reason |",
        "|---|---|---|---|---|---|",
    ]
    for b in flat:
        gt = COMBINED_GT.get(b["key"], "—")
        pred = speakers.get(b["key"], "—")
        r = reason.get(b["key"], "")
        tx = b["text"].replace("|", "\\|").replace("\n", " ")[:50]
        mark = ""
        if gt != "—":
            same = (
                gt == pred
                or (gt == "unknown" and pred in ("unknown", "other"))
                or (gt == "sfx" and pred in ("sfx", "noise"))
            )
            mark = " ✓" if same else " ✗"
        report.append(f"| `{b['key']}` | {b['page']} | {tx!r} | {gt} | {pred}{mark} | {r} |")

    (OUT / "probe_deterministic.md").write_text("\n".join(report), "utf-8")
    log.info("wrote %s", OUT / "probe_deterministic.md")


if __name__ == "__main__":
    asyncio.run(main())
