"""Block classification for Lens-detected text regions.

Each TextBlock is classified into one of three classes — sfx, dialogue,
narration — based on Lens geometry signals (rotation, aspect, char
count). The class drives ``BubbleGroup.shape_kind`` (which the
renderer reads for halo intensity and fit tolerance) and the erase
mask dilation profile.

Empirical thresholds tuned on the ch001 fixture survey (39 blocks):
    SFX (≤10 chars):        aspect 1.4–3.5, lines 1–2, words 1–3
    Short dialogue (11–30): aspect 0.5–2.7, lines 2–7, words 3–9
    Long dialogue (>30):    aspect 0.7–1.2, lines 5–7, words 9–15

Rotation > 5° is a strong SFX signal independent of size — only SFX
get angled-text typesetting in manga.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..contracts import TextBlock


__all__ = ["BlockClass", "ClassSignals", "PROFILES", "classify_block"]


BlockClass = Literal["sfx", "dialogue", "narration"]


# Char count is the primary axis because Lens typesets manga dialogue
# as 3–5 char lines (vertical strip style), making it more
# discriminative than line count.
_SFX_MAX_CHARS         = 10
_SFX_MIN_ASPECT        = 1.4
_SFX_ROTATION_OVERRIDE = 5.0     # > this tilt → SFX regardless of size
_SHORT_MAX_CHARS       = 30      # dialogue / narration boundary


@dataclass(frozen=True, slots=True)
class ClassSignals:
    """Per-class erase + render parameters tuned on real Lens data."""
    shape_kind:            str   # "burst" → glow halo; "dialogue" → plain
    erase_dilate_fraction: float
    erase_dilate_max_px:   int


PROFILES: dict[BlockClass, ClassSignals] = {
    # SFX: visually loud, wide angle range, often on busy art. Big
    # halo so glyphs survive screentone / gradient background.
    # shape_kind=burst tells render to use the glow stroke variant.
    "sfx": ClassSignals(
        shape_kind="burst",
        erase_dilate_fraction=0.08,
        erase_dilate_max_px=20,
    ),
    # Dialogue: 11–30 chars, in clean bubbles. Tight erase since the
    # bubble background is already solid white/black.
    "dialogue": ClassSignals(
        shape_kind="dialogue",
        erase_dilate_fraction=0.04,
        erase_dilate_max_px=14,
    ),
    # Narration: long blocks of caption text, usually on art
    # background (top / bottom of page, no bubble). Larger erase to
    # give the inpainter cleaner context between word rows.
    "narration": ClassSignals(
        shape_kind="dialogue",
        erase_dilate_fraction=0.06,
        erase_dilate_max_px=18,
    ),
}


def classify_block(block: TextBlock, text: str) -> BlockClass:
    """Pick SFX / dialogue / narration from Lens geometry signals.

    Rules, ordered by specificity:
      1. Rotation > 5° → SFX. Only SFX get angled-text typesetting in
         manga (survey: 4/4 angled blocks were SFX — BLUSH/ACK/ERR/SLUMP).
      2. Char count ≤ 10 AND wide aspect (w/h > 1.4) → SFX. Short text
         in a wide bbox is the canonical horizontal SFX pattern.
      3. Char count > 30 → narration. Long-form text doesn't fit in
         speech bubbles; it lives in captions.
      4. Default → dialogue.

    Lines / words are not used as primary discriminators because
    Lens line-splits vary per-page; the ch001 survey showed overlap
    between classes on those axes.
    """
    char_count = sum(1 for c in text if not c.isspace())
    x1, y1, x2, y2 = block.bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / h

    if abs(block.rotation_deg) > _SFX_ROTATION_OVERRIDE:
        return "sfx"
    if char_count <= _SFX_MAX_CHARS and aspect >= _SFX_MIN_ASPECT:
        return "sfx"
    if char_count > _SHORT_MAX_CHARS:
        return "narration"
    return "dialogue"
