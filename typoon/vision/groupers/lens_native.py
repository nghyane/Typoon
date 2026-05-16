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
    """1 Lens block = 1 BubbleGroup, with tategaki column merging.

    Japanese tategaki speech bubbles appear as multiple narrow vertical
    Lens blocks (one per column). Adjacent columns with high y-overlap
    are merged into a single BubbleGroup so the translator sees the full
    sentence and the renderer has a wider box to work with.
    """

    name = "lens_native"

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        return await asyncio.to_thread(self._build, detection, lang)

    def _build(
        self, detection: DetectionResult, lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        groups = [_block_to_group(b) for b in detection.blocks]
        return _merge_tategaki_columns(groups)


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
        text_direction=_infer_text_direction(block, typesetting),
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


def _is_cjk_text(text: str) -> bool:
    """Heuristic: text contains CJK ideograph / Hiragana / Katakana.

    Tategaki is essentially exclusive to Japanese and Chinese in manga;
    Korean (Hangul), Latin and Vietnamese are typeset horizontally even
    in vertical bubbles. Used as a secondary signal when bbox aspect is
    borderline (e.g. single-character columns where h/w < 2.0).
    """
    for c in text:
        cp = ord(c)
        if 0x3040 <= cp <= 0x30FF:        # Hiragana + Katakana
            return True
        if 0x4E00 <= cp <= 0x9FFF:        # CJK Unified Ideographs
            return True
        if 0x3400 <= cp <= 0x4DBF:        # CJK Extension A
            return True
        if 0xF900 <= cp <= 0xFAFF:        # CJK Compatibility Ideographs
            return True
    return False


def _infer_text_direction(
    block: TextBlock,
    typesetting: TypesettingHint | None,
) -> str:
    """Infer source text_direction from Lens block geometry + script.

    Two signals, in order:

      1. Strict aspect: ``h > w * 2.0`` — unambiguous tategaki column.
         Empirical survey on chapter 112 (10 JP blocks): aspect h/w
         ranges from 3.4 to 10.7, all clearly above 2.0.
      2. CJK script + ``h > w`` — catches short 1–2 character columns
         (e.g. 「滚」, 「前輩」) where strict aspect misses. JP/CN are the
         dominant tategaki sources; Korean / Latin / Vietnamese in
         manga are written horizontally, so script alone is a reliable
         secondary signal when aspect is borderline.

    For non-CJK horizontal prose the bbox is usually wider than tall so
    neither rule fires.
    """
    x1, y1, x2, y2 = block.bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if h > w * 2.0:
        return "vertical"
    if h > w and _is_cjk_text(block.text or ""):
        return "vertical"
    return "horizontal"


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
      4. Row-gap recovery: when a non-edge line is anomalously short
         vs the rest of the block (Lens dropped glyphs around dense
         decoration runs such as 「······」), paint the full block-width
         band at that line's y-range. The detector may have re-OCRed
         the row to recover the source text, but the word bboxes still
         only cover the originally-recognised glyphs — so erase still
         needs the broader band to wipe the dropped glyphs from canvas.
      5. Dilate base by font-size proportional radius.
      6. Clip dilated AND in_bounds → final mask. The clip guarantees
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

    # Row-gap recovery: stretch suspicious row band to full block width
    for idx in _suspicious_line_indices(block):
        line = block.lines[idx]
        ly1 = max(0, min(h, line.bbox[1] - y1))
        ly2 = max(0, min(h, line.bbox[3] - y1))
        if ly2 <= ly1:
            continue
        base[ly1:ly2, :] = 255
        ib_y1 = max(0, ly1 - pad_y)
        ib_y2 = min(h, ly2 + pad_y)
        in_bounds[ib_y1:ib_y2, :] = 255

    # Dilate the glyph base by font-proportional radius, then clip back
    # inside the expanded supports. Result: tight around glyphs, never
    # leaks past per-word headroom.
    dilate_radius = max(1, int(font_size_px * 0.10))
    ksize = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(base, kernel, iterations=1)
    refined = np.where(in_bounds > 0, dilated, 0).astype(np.uint8)

    return TextMask(x=x1, y=y1, image=refined), False


# Row recognition-gap detector. Lens occasionally drops glyphs around
# dense decoration runs (e.g. CJK ellipsis 「······」) and emits only the
# unaffected suffix for that row; the row bbox stays correct so a
# full-width band still localises the missed glyphs for the erase mask.
# Mirrors `_suspicious_line_indices` in `lens_blocks.py` — kept local to
# preserve the cli/stages/adapters/runs/cli dependency boundary
# (grouper must not depend on detector internals).
_ROW_GAP_SHORT_RATIO  = 0.5
_ROW_GAP_MIN_LINES    = 3


def _suspicious_line_indices(block: TextBlock) -> list[int]:
    lines = block.lines
    if len(lines) < _ROW_GAP_MIN_LINES:
        return []
    widths = [max(1, l.bbox[2] - l.bbox[0]) for l in lines]
    out: list[int] = []
    for i in range(1, len(lines) - 1):
        others = widths[:i] + widths[i + 1:]
        median_w = statistics.median(others)
        if widths[i] / median_w < _ROW_GAP_SHORT_RATIO:
            out.append(i)
    return out


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


# ─── Tategaki column merging ──────────────────────────────────────────────

# Tategaki columns belonging to the same speech bubble share nearly the
# same vertical span and sit side-by-side. We cluster vertical blocks by
# chaining right-to-left along x (reading order) with a self-calibrating
# x-gap budget tied to column width, then apply guards:
#
#   - cluster_guard_font   — drop merge if max(font_px)/min(font_px) > 1.8
#                            (cross-bubble cluster: different bubbles use
#                             different font sizes)
#   - cluster_guard_outsider — drop merge if any non-cluster vertical
#                              column has its centre inside the cluster
#                              bbox (cluster crossed a bubble boundary)
#   - _MAX_COLUMNS         — safety cap on cluster size
#
# Columns flow right-to-left in tategaki. After merging, the text is
# sorted right-to-left (descending x) so reading order is correct.

_Y_OVERLAP_MIN       = 0.50   # fraction of shorter column that must overlap
_X_GAP_FLOOR_PX      = 80     # absolute min gap budget (low-res safety)
_X_GAP_WIDTH_MULT    = 2.0    # gap budget scales with min(column_width)
_FONT_RATIO_MAX      = 1.8    # max(font_px)/min(font_px) inside a cluster
_MAX_COLUMNS         = 6      # safety cap on cluster size


def _merge_tategaki_columns(
    groups: list[BubbleGroup],
) -> tuple[BubbleGroup, ...]:
    """Chain-cluster adjacent tategaki columns into single bubbles.

    Algorithm:
      1. Collect vertical-direction columns, sort right-to-left (manga
         tategaki reading order).
      2. For each column in order, evaluate every open cluster; join
         the cluster with the strongest y-overlap that also passes the
         x-gap budget (``_compatible_with_cluster``). If none match,
         open a new cluster. Evaluating all open clusters (not just
         the last) handles interleaving — a column from a different
         bubble sitting between two clusters by x doesn't fork the
         right one.
      3. After clustering, drop merges that fail either guard:
           - font-size ratio inside cluster too large
           - another vertical column's centre lies inside cluster bbox
         Failed clusters fall back to their original 1-column groups.
      4. Build merged BubbleGroup per surviving cluster; preserve
         original top-to-bottom page order in the final output.
    """
    if not groups:
        return ()

    vertical_idx  = [i for i, g in enumerate(groups) if g.text_direction == "vertical"]
    other_groups  = [g for g in groups if g.text_direction != "vertical"]
    vertical = [groups[i] for i in vertical_idx]

    if not vertical:
        return tuple(groups)

    # Right-to-left sweep over x-centres. For each column, try every
    # open cluster and join the one with the strongest y-overlap that
    # also passes the x-gap budget. This handles cases where columns
    # from a different bubble interleave by x but not by y.
    order = sorted(range(len(vertical)), key=lambda i: -_x_centre(vertical[i].bbox))

    clusters: list[list[int]] = []
    for local in order:
        col = vertical[local]
        best_idx: int | None = None
        best_score = -1.0
        for ci, cluster in enumerate(clusters):
            members = [vertical[i] for i in cluster]
            if not _compatible_with_cluster(col, members):
                continue
            score = max(_y_overlap_ratio(col.bbox, m.bbox) for m in members)
            if score > best_score:
                best_score = score
                best_idx = ci
        if best_idx is not None:
            clusters[best_idx].append(local)
        else:
            clusters.append([local])

    # Apply cluster guards. A failed cluster splits back into singletons.
    final_clusters: list[list[int]] = []
    for cluster in clusters:
        members = [vertical[i] for i in cluster]
        if len(cluster) == 1:
            final_clusters.append(cluster)
            continue
        if len(cluster) > _MAX_COLUMNS:
            final_clusters.extend([[i] for i in cluster])
            continue
        if not _passes_font_guard(members):
            final_clusters.extend([[i] for i in cluster])
            continue
        if not _passes_outsider_guard(members, vertical, cluster):
            final_clusters.extend([[i] for i in cluster])
            continue
        final_clusters.append(cluster)

    # Build output
    result: list[BubbleGroup] = list(other_groups)
    for cluster in final_clusters:
        members = [vertical[i] for i in cluster]
        if len(members) == 1:
            result.append(members[0])
        else:
            result.append(_merge_group_list(members))

    # Preserve original top-to-bottom page order
    result.sort(key=lambda g: g.bbox[1])
    return tuple(result)


def _x_centre(bbox: tuple[int, int, int, int]) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _y_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ha = max(1, a[3] - a[1])
    hb = max(1, b[3] - b[1])
    y_top = max(a[1], b[1])
    y_bot = min(a[3], b[3])
    overlap = max(0, y_bot - y_top)
    return overlap / min(ha, hb)


def _x_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    """Edge-to-edge horizontal gap; negative if columns overlap on x."""
    return max(a[0], b[0]) - min(a[2], b[2])


def _compatible_with_cluster(
    candidate: BubbleGroup,
    members: list[BubbleGroup],
) -> bool:
    """True if candidate can join an existing cluster.

    Required:
      * y-overlap ≥ _Y_OVERLAP_MIN with at least one existing member
        (not the union bbox — a tall outlier would otherwise dominate).
      * edge-to-edge x-gap from the leftmost member ≤ width-scaled budget.
        Gap is measured against the leftmost member because we sweep
        right-to-left; the chain extends left.
    """
    if not any(
        _y_overlap_ratio(candidate.bbox, m.bbox) >= _Y_OVERLAP_MIN
        for m in members
    ):
        return False

    leftmost = min(members, key=lambda m: m.bbox[0])
    wa = max(1, candidate.bbox[2] - candidate.bbox[0])
    wb = max(1, leftmost.bbox[2] - leftmost.bbox[0])
    gap_cap = max(_X_GAP_FLOOR_PX, int(min(wa, wb) * _X_GAP_WIDTH_MULT))
    gap = _x_gap(candidate.bbox, leftmost.bbox)
    return -2 <= gap <= gap_cap  # -2 tolerates 1px overlap (rounding)


def _passes_font_guard(members: list[BubbleGroup]) -> bool:
    """Reject clusters whose columns disagree on glyph size.

    Same-bubble columns share a glyph size; cross-bubble columns
    typically don't. The size signal differs by direction:

      * Vertical (tategaki): glyph width ≈ column bbox width. We avoid
        ``typesetting.font_size_px`` here because Lens reports one "line"
        per column, so its line height equals column height (~ glyph
        size × char count), not glyph size.
      * Horizontal: ``typesetting.font_size_px`` is the median line
        height — a direct font-size proxy.

    Surveyed JP/CN manga: same-bubble glyph sizes are within ~10% of
    each other; cross-bubble pairs typically diverge ≥ 1.8×.
    """
    sizes: list[float] = []
    for m in members:
        if m.text_direction == "vertical":
            w = m.bbox[2] - m.bbox[0]
            if w > 0:
                sizes.append(float(w))
        elif m.typesetting is not None and m.typesetting.font_size_px > 0:
            sizes.append(float(m.typesetting.font_size_px))
    if len(sizes) < 2:
        return True
    return max(sizes) / min(sizes) <= _FONT_RATIO_MAX


def _passes_outsider_guard(
    members: list[BubbleGroup],
    all_vertical: list[BubbleGroup],
    member_indices: list[int],
) -> bool:
    """Reject clusters whose bbox swallows another vertical column.

    If a column belongs to a different bubble but its centre falls inside
    our cluster's bbox, the cluster crossed a bubble boundary. The bbox
    only spans member columns, so any non-member centre inside it is a
    structural conflict.
    """
    member_set = set(member_indices)
    x1 = min(m.bbox[0] for m in members)
    y1 = min(m.bbox[1] for m in members)
    x2 = max(m.bbox[2] for m in members)
    y2 = max(m.bbox[3] for m in members)
    for idx, col in enumerate(all_vertical):
        if idx in member_set:
            continue
        cx = _x_centre(col.bbox)
        cy = (col.bbox[1] + col.bbox[3]) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return False
    return True


def _merge_group_list(groups: list[BubbleGroup]) -> BubbleGroup:
    """Merge multiple tategaki column BubbleGroups into one."""
    # Union bbox
    x1 = min(g.bbox[0] for g in groups)
    y1 = min(g.bbox[1] for g in groups)
    x2 = max(g.bbox[2] for g in groups)
    y2 = max(g.bbox[3] for g in groups)
    merged_bbox = (x1, y1, x2, y2)

    # Text: sort columns right-to-left (tategaki reading order) then join
    sorted_cols = sorted(groups, key=lambda g: -g.bbox[0])
    merged_text = "\n".join(g.text for g in sorted_cols if g.text.strip())

    # Masks: union of all column masks
    all_text_masks  = tuple(m for g in groups for m in g.text_masks)
    all_erase_masks = tuple(m for g in groups for m in g.erase_masks)

    # Typesetting: take the one with the most lines (best geometry signal)
    best_ts = max(
        (g.typesetting for g in groups if g.typesetting is not None),
        key=lambda ts: ts.line_count,
        default=None,
    )
    # For a merged bubble, avg_chars_per_line reflects the full sentence
    if best_ts is not None and merged_text:
        total_chars = sum(1 for c in merged_text if not c.isspace())
        total_lines = sum(g.typesetting.line_count for g in groups if g.typesetting)
        if total_lines > 0:
            from ..contracts import TypesettingHint
            best_ts = TypesettingHint(
                font_size_px=best_ts.font_size_px,
                line_count=total_lines,
                avg_chars_per_line=total_chars / total_lines,
            )

    confidence = max(g.confidence for g in groups)

    return BubbleGroup(
        bbox=merged_bbox,
        polygon=_bbox_to_polygon(merged_bbox),
        text=merged_text,
        confidence=confidence,
        text_masks=all_text_masks,
        erase_masks=all_erase_masks,
        source="lens",
        shape_kind=groups[0].shape_kind,
        used_fallback=any(g.used_fallback for g in groups),
        rotation_deg=groups[0].rotation_deg,
        typesetting=best_ts,
        text_direction="vertical",
    )
