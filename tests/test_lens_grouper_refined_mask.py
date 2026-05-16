"""Refined word-union mask tests.

The grouper builds a koharu-style mask: dilate the union of word
bboxes, then clip back into a per-word expanded support region so
the mask never leaks into neighbouring art. These tests pin that
behaviour.
"""

from __future__ import annotations

import asyncio

import numpy as np

from typoon.vision.contracts import (
    DetectionResult,
    LineBox,
    TextBlock,
    WordBox,
)
from typoon.vision.groupers.lens_native import LensNativeGrouper


def _block_with_lines(bbox, text, *, words, lines, rotation_deg=0.0):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        words=tuple(words), lines=tuple(lines), rotation_deg=rotation_deg,
    )


def _run(detection):
    grouper = LensNativeGrouper()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    return asyncio.run(grouper.group(image, detection, "en"))


# ─── Refined mask grows around glyphs ─────────────────────────────────────


def test_refined_mask_extends_beyond_raw_word_bboxes():
    """Dilated mask should cover pixels just outside the word bbox
    (within the expanded support). Letters often spill past their
    OCR bbox by a few pixels for ascenders/descenders."""
    block = _block_with_lines(
        (100, 100, 300, 200),
        "hello",
        words=[WordBox(bbox=(110, 110, 250, 190), text="hello")],
        # font_size = line height = 80
        lines=[LineBox(bbox=(100, 110, 300, 190), text="hello")],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]

    # Inside word bbox: definitely painted
    # (110-100=10, 110-100=10) → (140-100=40, 140-100=40) → block-local (10,10)..(40,40)
    assert tm.image[20, 20] == 255

    # Just outside word bbox but inside expanded support: should be
    # painted by dilation (font_size=80 × 0.10 dilate ≈ 8px)
    # Word bbox ends at lx=150 in block-local; check at lx=155
    assert tm.image[40, 155] == 255, (
        "dilation should extend a few pixels past the word bbox"
    )


def test_refined_mask_does_not_leak_to_block_corners():
    """The expanded support is per-word, not per-block, so empty corners
    of the block (no nearby word) must stay 0."""
    block = _block_with_lines(
        (0, 0, 400, 200),
        "hi",
        # Tiny word in the centre
        words=[WordBox(bbox=(180, 90, 220, 110), text="hi")],
        lines=[LineBox(bbox=(180, 90, 220, 110), text="hi")],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    # Block corner should be 0 (far from any word)
    assert tm.image[5, 5] == 0
    assert tm.image[5, 395] == 0
    assert tm.image[195, 5] == 0


def test_refined_mask_clips_within_expanded_support():
    """Even with aggressive dilate, mask cannot bleed past the
    per-word expanded support."""
    block = _block_with_lines(
        (0, 0, 400, 200),
        "x",
        # Single small word at left edge
        words=[WordBox(bbox=(10, 90, 30, 110), text="x")],
        # Tiny font size → tiny expand pad
        lines=[LineBox(bbox=(10, 90, 30, 110), text="x")],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    # Far right of block: must NOT be painted (expanded support is
    # only ~4-6px around the tiny word)
    assert tm.image[100, 200] == 0
    assert tm.image[100, 300] == 0


def test_refined_mask_fills_per_word_padding():
    """Expand pad is font-size proportional. Larger font → bigger
    halo around each word."""
    big_font_block = _block_with_lines(
        (0, 0, 400, 200),
        "big",
        words=[WordBox(bbox=(100, 50, 300, 150), text="big")],
        lines=[LineBox(bbox=(100, 50, 300, 150), text="big")],  # font ≈ 100
    )
    det = DetectionResult(
        blocks=(big_font_block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    # Word ends at lx=300; with font=100 and pad_x=12% → expanded to lx=312
    # plus dilate radius 10 → still painted at lx~308
    # (in block-local coords; word at 100..300 → block-local 100..300)
    assert tm.image[100, 305] == 255


# ─── Row-gap aware mask ───────────────────────────────────────────────────


def test_row_gap_mask_stretches_short_middle_row():
    """When a non-edge row is anomalously narrow vs the others, the
    mask must paint the full block width at that row's y-range.

    This covers the case where Lens drops glyphs around dense
    decoration (e.g. CJK ellipsis 「······」). The block geometry is
    fine but the word bboxes only cover the surviving suffix; erase
    must wipe the dropped glyphs from canvas too.
    """
    # 4-row block: rows 0, 1, 3 are 200 wide; row 2 only covers the
    # right-most word band (x=300..360). Median of others = 200 → row 2
    # ratio 60/200 = 0.30 < 0.5 → flagged.
    block = _block_with_lines(
        (0, 0, 400, 200),
        "row0 row1 short row3",
        words=[
            WordBox(bbox=(20, 5,   220, 35),  text="row0"),
            WordBox(bbox=(20, 55,  220, 85),  text="row1"),
            # row 2 only has a tail word at the far right
            WordBox(bbox=(300, 105, 360, 135), text="short"),
            WordBox(bbox=(20, 155, 220, 185), text="row3"),
        ],
        lines=[
            LineBox(bbox=(20, 5,   220, 35),  text="row0"),
            LineBox(bbox=(20, 55,  220, 85),  text="row1"),
            LineBox(bbox=(300, 105, 360, 135), text="short"),
            LineBox(bbox=(20, 155, 220, 185), text="row3"),
        ],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    # In block-local coords. Row 2 y-range is 105..135.
    # Without row-gap recovery, x ∈ [10, 290] in that band would be 0.
    # With recovery, the whole band x ∈ [0, 400) should be painted.
    assert tm.image[120, 10]  == 255, "row 2 left half should be painted"
    assert tm.image[120, 200] == 255, "row 2 middle should be painted"
    assert tm.image[120, 350] == 255, "row 2 right (tail word) painted"
    # Rows 1 and 3 should NOT bleed sideways — verify a point far from
    # any word but in their y-band is still 0 (mask did not become a
    # full block rect).
    # Row 1 y=55..85. Mid x=300 is outside word bbox, far from row 0/2.
    # Expanded pad ~12% × font (=30) ≈ 4 px; with dilate 3, total reach
    # is small. Pixel at row1_y=70, x=295 must be 0.
    assert tm.image[70, 295] == 0, "row 1 right gap stayed clean"


def test_row_gap_mask_does_not_trigger_on_edge_rows():
    """First and last rows being short is legitimate. Mask must not
    stretch the edges to block width when only edge rows are narrow."""
    block = _block_with_lines(
        (0, 0, 400, 200),
        "x mid mid x",
        words=[
            WordBox(bbox=(20, 5,   80,  35),  text="x"),     # short edge
            WordBox(bbox=(20, 55,  380, 85),  text="middle 1"),
            WordBox(bbox=(20, 105, 380, 135), text="middle 2"),
            WordBox(bbox=(20, 155, 80,  185), text="x"),     # short edge
        ],
        lines=[
            LineBox(bbox=(20, 5,   80,  35),  text="x"),
            LineBox(bbox=(20, 55,  380, 85),  text="middle 1"),
            LineBox(bbox=(20, 105, 380, 135), text="middle 2"),
            LineBox(bbox=(20, 155, 80,  185), text="x"),
        ],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    # Row 0 right of word (x=200, y=20) — must stay 0. If edge-row
    # stretching kicked in this would be 255.
    assert tm.image[20, 300]  == 0, "edge row 0 right gap stayed clean"
    assert tm.image[170, 300] == 0, "edge row 3 right gap stayed clean"
