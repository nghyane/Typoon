"""Lens-native grouper.

Each Lens block maps 1:1 to a BubbleGroup with Lens-derived metadata
threaded through to render. Three concerns live here:

  1. Glyph-precise erase mask via per-word bboxes (or block-rect fallback).
  2. Typesetting hint (font_size / line_count / aspect) for render fit.
  3. Block-class detection — SFX vs dialogue vs narration — from Lens
     geometry signals (aspect, char count, line count, rotation). This
     drives shape_kind (which the renderer reads for halo intensity and
     fit tolerance).

Classification rules are calibrated on real Lens output across the
fixture chapters. See `_classify_block` for the empirical thresholds.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ..contracts import (
    BubbleGroup,
    DetectionResult,
    LineBox,
    TextBlock,
    TextMask,
    TypesettingHint,
)


__all__ = ["LensNativeGrouper"]


# ─── Empirical thresholds (Lens output, ch001 fixture survey) ─────────────


# Class detection bands. Char count is the primary axis because Lens
# typesets manga dialogue as 3-5 char lines (vertical strip style),
# making char count more discriminative than line count.
#
# Survey on 39 ch001 blocks:
#   SFX (≤10 chars):       aspect 1.4-3.5, lines 1-2,  words 1-3
#   Short dialogue (11-30): aspect 0.5-2.7, lines 2-7, words 3-9
#   Long dialogue (>30):    aspect 0.7-1.2, lines 5-7, words 9-15
#
# Rotation > 5° is a strong SFX signal independent of size — only SFX
# get angled-text typesetting in manga.
_SFX_MAX_CHARS  = 10
_SFX_MIN_ASPECT = 1.4
_SFX_ROTATION_OVERRIDE = 5.0   # any block tilted > 5° is SFX regardless of size

_SHORT_MAX_CHARS = 30          # dialogue boundary; > this = long narration


BlockClass = Literal["sfx", "dialogue", "narration"]


@dataclass(frozen=True, slots=True)
class _ClassSignals:
    """Per-class erase + render parameters tuned on real Lens data."""
    shape_kind:                str   # "burst" maps to glow halo; "dialogue" plain
    erase_dilate_fraction:     float
    erase_dilate_max_px:       int


_PROFILES: dict[BlockClass, _ClassSignals] = {
    # SFX: visually loud, wide angle range, often on busy art. Big halo
    # so glyphs survive screentone / gradient background. shape_kind=burst
    # tells render to use the glow stroke variant.
    "sfx": _ClassSignals(
        shape_kind="burst",
        erase_dilate_fraction=0.08,
        erase_dilate_max_px=20,
    ),
    # Dialogue: 11-30 chars, in clean bubbles. Tight erase since the
    # bubble background is already solid white/black.
    "dialogue": _ClassSignals(
        shape_kind="dialogue",
        erase_dilate_fraction=0.04,
        erase_dilate_max_px=14,
    ),
    # Narration: long blocks of caption text, usually on art background
    # (top/bottom of page, no bubble). Larger erase to give the cleaner
    # context to inpaint between word rows.
    "narration": _ClassSignals(
        shape_kind="dialogue",
        erase_dilate_fraction=0.06,
        erase_dilate_max_px=18,
    ),
}


# ─── Public grouper ───────────────────────────────────────────────────────


class LensNativeGrouper:
    """1 Lens block = 1 BubbleGroup. Pure assembly, no merging."""

    name = "lens_native"

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        return await asyncio.to_thread(self._build, detection)

    def _build(self, detection: DetectionResult) -> tuple[BubbleGroup, ...]:
        return tuple(_block_to_group(b) for b in detection.blocks)


# ─── Block → group ────────────────────────────────────────────────────────


def _block_to_group(block: TextBlock) -> BubbleGroup:
    text = block.text or ""
    block_class = _classify_block(block, text)
    profile = _PROFILES[block_class]
    typesetting = _build_typesetting_hint(block)
    font_size_px = typesetting.font_size_px if typesetting else 0

    glyph_mask, used_fallback = _build_glyph_mask(block, font_size_px)
    erase_mask = _dilate_mask(
        glyph_mask,
        _erase_dilate_px(block.bbox, profile, font_size_px),
    )

    return BubbleGroup(
        bbox=block.bbox,
        polygon=_bbox_to_polygon(block.bbox),
        text=text,
        confidence=block.confidence,
        text_masks=(glyph_mask,),
        erase_masks=(erase_mask,),
        source="lens",
        shape_kind=profile.shape_kind,
        used_fallback=used_fallback,
        rotation_deg=block.rotation_deg,
        typesetting=typesetting,
    )


# ─── Block classification ─────────────────────────────────────────────────


def _classify_block(block: TextBlock, text: str) -> BlockClass:
    """Pick SFX / dialogue / narration from Lens geometry signals.

    Rules, ordered by specificity:

      1. Rotation > 5° → SFX. Only SFX get angled-text typesetting in
         manga. Survey: 4/4 angled blocks were SFX (BLUSH/ACK/ERR/SLUMP).
      2. Char count ≤ 10 AND wide aspect (w/h > 1.4) → SFX. Short text
         in a wide bubble is the canonical horizontal SFX pattern.
      3. Char count > 30 → narration. Long form text doesn't fit in
         speech bubbles; it lives in captions.
      4. Default → dialogue.

    Lines / words are not used as primary discriminators because Lens
    line-splits vary per-page; survey showed overlap between classes
    on those axes.
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


# ─── Typesetting hint ─────────────────────────────────────────────────────


def _build_typesetting_hint(block: TextBlock) -> TypesettingHint | None:
    lines = block.lines
    if not lines:
        return None

    # Median line height = original font size in page pixels. Median
    # (not mean) because Lens occasionally emits a tall outlier for the
    # last line when it includes punctuation tails.
    heights = [max(1, l.bbox[3] - l.bbox[1]) for l in lines]
    font_px = int(statistics.median(heights))

    char_counts = [
        sum(1 for c in l.text if not c.isspace()) for l in lines
    ]
    total_chars = sum(char_counts)
    avg_chars = total_chars / len(lines) if lines else 0.0

    return TypesettingHint(
        font_size_px=max(1, font_px),
        line_count=len(lines),
        avg_chars_per_line=avg_chars,
    )


# ─── Erase mask construction ──────────────────────────────────────────────


def _build_glyph_mask(
    block: TextBlock, font_size_px: int,
) -> tuple[TextMask, bool]:
    """Word-union mask refined inside per-word expanded support.

    Algorithm (mirrors koharu.refine_segmentation_mask, adapted for Lens
    word-bbox input — no per-pixel probability map available):

      1. For each word, compute an expanded support rect (font-size
         proportional padding). This is the region where dilation is
         allowed to live for that word.
      2. Union all expanded supports → headroom mask `in_bounds`.
      3. Union all word bboxes (the actual glyph pixels) → base mask.
      4. Dilate base by font-size proportional radius.
      5. Clip dilated AND in_bounds → final mask. The clip guarantees
         we never bleed past the per-word headroom even if dilate
         pushes far on a tall diacritic.

    Falls back to block-rect mask when words are unavailable.
    """
    if not block.words:
        return _block_rect_mask(block.bbox), True

    x1, y1, x2, y2 = block.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    # Compute expanded supports (in block-local coords)
    pad_x = max(_EXPAND_PAD_MIN_PX, int(font_size_px * _EXPAND_PAD_X_FRACTION))
    pad_y = max(_EXPAND_PAD_MIN_PX, int(font_size_px * _EXPAND_PAD_Y_FRACTION))

    base = np.zeros((h, w), dtype=np.uint8)
    in_bounds = np.zeros((h, w), dtype=np.uint8)
    any_painted = False

    for word in block.words:
        wx1, wy1, wx2, wy2 = word.bbox
        # Glyph base — word bbox clipped to block bounds
        lx1 = max(0, min(w, wx1 - x1))
        ly1 = max(0, min(h, wy1 - y1))
        lx2 = max(0, min(w, wx2 - x1))
        ly2 = max(0, min(h, wy2 - y1))
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        base[ly1:ly2, lx1:lx2] = 255
        any_painted = True

        # Expanded support — word bbox + font-proportional padding,
        # clipped to block bounds. This is the dilation budget.
        ex1 = max(0, min(w, wx1 - x1 - pad_x))
        ey1 = max(0, min(h, wy1 - y1 - pad_y))
        ex2 = max(0, min(w, wx2 - x1 + pad_x))
        ey2 = max(0, min(h, wy2 - y1 + pad_y))
        in_bounds[ey1:ey2, ex1:ex2] = 255

    if not any_painted:
        return _block_rect_mask(block.bbox), True

    # Dilate the glyph base by font-proportional radius, then clip back
    # inside the expanded supports. Result: tight around glyphs, never
    # leaks past per-word headroom.
    dilate_radius = max(1, int(font_size_px * 0.10))
    ksize = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(base, kernel, iterations=1)
    refined = np.where(in_bounds > 0, dilated, 0).astype(np.uint8)

    return TextMask(x=x1, y=y1, image=refined), False


def _block_rect_mask(bbox: tuple[int, int, int, int]) -> TextMask:
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    image = np.full((h, w), 255, dtype=np.uint8)
    return TextMask(x=x1, y=y1, image=image)


_ERASE_DILATE_MIN_PX = 3

# Font-size proportional padding for the per-word/line expanded support
# region (mirrors koharu.refine_segmentation_mask.expanded_text_block_crop_bounds).
# This is the room around each word's bbox where the dilated mask is
# allowed to live; the dilate result is clipped back inside it so the
# mask never bleeds onto neighbouring art.
_EXPAND_PAD_X_FRACTION = 0.12   # horizontal padding (~12% font size)
_EXPAND_PAD_Y_FRACTION = 0.18   # vertical padding (~18% font size) — diacritics
_EXPAND_PAD_MIN_PX     = 2


def _erase_dilate_px(
    bbox: tuple[int, int, int, int],
    profile: _ClassSignals,
    font_size_px: int,
) -> int:
    """Dilate radius for the erase mask.

    Scales with detected font size when available (preferred — gives
    consistent halo regardless of bbox aspect), falls back to bbox short
    side. The font signal comes from Lens line geometry via TypesettingHint.
    """
    base = font_size_px if font_size_px > 0 else min(
        bbox[2] - bbox[0], bbox[3] - bbox[1],
    )
    base = max(1, base)
    return int(max(
        _ERASE_DILATE_MIN_PX,
        min(base * profile.erase_dilate_fraction, profile.erase_dilate_max_px),
    ))


def _dilate_mask(mask: TextMask, pad: int) -> TextMask:
    if pad <= 0:
        return mask
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(expanded, kernel, iterations=1)
    return TextMask(x=mask.x - pad, y=mask.y - pad, image=dilated)


def _bbox_to_polygon(
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = bbox
    return (
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    )
