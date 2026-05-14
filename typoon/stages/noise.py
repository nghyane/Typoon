"""Deterministic noise classifier — pure regex, no LLM.

A bubble is "noise" (non-diegetic) if it's unambiguously platform chrome,
a watermark, a page counter, or pure OCR rubble. The patterns live in
`skills_data/noise_terms.txt` and are intentionally narrow — false
positives silently drop story dialogue.

Two consumers:

  - `stages.brief.build_chapter_brief`: folds deterministic noise into
    `ChapterBrief.noise_keys` before the vision pass even sees these
    bubbles. They are filtered out of the translator pipeline upstream.

  - `stages.page._parse_blocks`: defense-in-depth — if the model
    nonetheless emits a block for a deterministic-noise bubble, we
    rewrite it to `kind="skip"` so render does the right thing.

Keep this module dependency-free relative to `brief.py` and `page.py`
so neither side has to import the other (circular import risk).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_OCR_NOISE_RE = re.compile(r"^[\W_\d]+$")
_NOISE_TERMS_PATH = Path(__file__).with_name("skills_data") / "noise_terms.txt"


@lru_cache(maxsize=1)
def _noise_term_patterns() -> tuple[re.Pattern[str], ...]:
    """Load and compile deterministic noise regexes once.

    Patterns live in skills_data/noise_terms.txt — one regex per line,
    matched case-insensitively against stripped bubble text. Comment
    lines start with '#'.
    """
    if not _NOISE_TERMS_PATH.exists():
        return ()
    out: list[re.Pattern[str]] = []
    for raw in _NOISE_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(re.compile(line, re.IGNORECASE))
        except re.error:
            continue
    return tuple(out)


def is_auto_skip(text: str) -> bool:
    """True if `text` is deterministically non-diegetic and should be
    skipped without an LLM round trip.

    The decision is conservative: empty text, single digits/letters,
    pure punctuation/digit rubble (unless containing currency / time
    separators), and anything matching `noise_terms.txt` patterns.
    """
    s = text.strip()
    if not s:
        return True
    if s.isdigit() or (len(s) == 1 and s.isalpha()):
        return True
    if _OCR_NOISE_RE.fullmatch(s) and not any(ch in s for ch in ":/%$¥₩€£"):
        return True
    for pat in _noise_term_patterns():
        if pat.fullmatch(s):
            return True
    return False
