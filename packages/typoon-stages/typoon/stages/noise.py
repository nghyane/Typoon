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

`strip_noise_tokens` is a separate concern: it removes inline noise tokens
(watermark domains, emoji-logos, scanlation tags) that are appended to
otherwise-valid dialogue. Call it on OCR text before classification or
translation — it preserves the story content while discarding the noise.

Keep this module dependency-free relative to `brief.py` and `page.py`
so neither side has to import the other (circular import risk).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_OCR_NOISE_RE     = re.compile(r"^[\W_\d]+$")
_NOISE_TERMS_PATH = Path(__file__).with_name("skills_data") / "noise_terms.txt"

# ── Inline noise token patterns ────────────────────────────────────────────
# These match noise tokens that appear INSIDE otherwise-valid dialogue.
# strip_noise_tokens() removes them without touching the surrounding text.

# Domain / URL suffixes appended by OCR (e.g. "...影响。baozimm.com")
_INLINE_DOMAIN_RE = re.compile(
    r'\b[a-z0-9][a-z0-9.\-]{1,30}'
    r'\.(?:co|com|net|org|io|to|me|xyz|info|live|app|tw|hk|cn)'
    r'(?:/\S*)?\b',
    re.IGNORECASE,
)

# Lone emoji that are publisher logos (single emoji, not part of dialogue)
# e.g. 🥟 (baozi logo), 🐱 (cat logo), etc. — remove only when at boundary
_LOGO_EMOJI_RE = re.compile(
    r'(?<![^\s！？。，、])'   # preceded by whitespace or CJK punctuation or start
    r'[\U0001F300-\U0001F9FF]'  # any emoji in the common emoji range
    r'(?![^\s！？。，、])',    # followed by whitespace or CJK punctuation or end
)

# Scanlation credit tags that can be appended: [Reset-Scan] «baozimh» etc.
_CREDIT_TAG_RE = re.compile(
    r'[\[《«【「\(]\s*[a-z0-9][a-z0-9\s.\-]{1,30}\s*[\]》»】」\)]',
    re.IGNORECASE,
)


def strip_noise_tokens(text: str) -> str:
    """Remove inline noise tokens from otherwise-valid dialogue text.

    Strips:
    - Domain names / URLs appended by OCR (baozimm.com, sfacg.com, …)
    - Lone publisher-logo emoji at word boundaries
    - Scanlation credit tags [Tag], 《Tag》, etc.

    Does NOT strip Chinese/Japanese/Korean characters, punctuation, or
    anything that could be story content. Returns stripped text with
    leading/trailing whitespace cleaned up.
    """
    s = _INLINE_DOMAIN_RE.sub("", text)
    s = _LOGO_EMOJI_RE.sub("", s)
    s = _CREDIT_TAG_RE.sub("", s)
    # Collapse double spaces / punctuation gaps left by removal
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()


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
