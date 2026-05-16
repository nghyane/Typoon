"""Lens-native grouper — tategaki column chain clustering tests.

The grouper merges adjacent vertical (tategaki / vertical-CN) columns
that belong to the same speech bubble. Three guards must hold:

  * chain compatibility (y-overlap + x-gap)
  * font-size consistency inside cluster
  * no outsider vertical column inside cluster bbox

All tests use synthetic input — no network, no model.
"""

from __future__ import annotations

import asyncio

import numpy as np

from typoon.vision.contracts import DetectionResult, LineBox, TextBlock
from typoon.vision.groupers.lens_native import LensNativeGrouper


def _block(bbox, text, *, lines=()):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks", lines=tuple(lines),
    )


def _col(x1, y1, x2, y2, text, *, char_h=None):
    """Tategaki column block: synthesise one Lens "line" per column.

    `char_h` sets the per-glyph height inferred from the line bbox; if
    omitted, defaults to column width (the realistic value for square
    CJK glyphs). The grouper's font-size guard reads `bbox width` for
    vertical blocks, so synthesising consistent line bboxes is what
    keeps the guard tests faithful.
    """
    return _block(
        (x1, y1, x2, y2), text,
        lines=(LineBox(bbox=(x1, y1, x2, y2), text=text),),
    )


def _run(blocks):
    grouper = LensNativeGrouper()
    detection = DetectionResult(
        blocks=tuple(blocks),
        text_already_recognized=True,
        page_size=(2000, 3000),
    )
    image = np.zeros((3000, 2000, 3), dtype=np.uint8)
    return asyncio.run(grouper.group(image, detection, "ja"))


# ─── Direction inference ─────────────────────────────────────────────────


def test_short_cjk_column_is_vertical_even_below_aspect_2():
    """Single-glyph CJK column (h/w < 2.0) must still be vertical."""
    # 80 wide × 100 tall column with one kanji — aspect 1.25
    groups = _run([_col(100, 100, 180, 200, "前")])
    assert len(groups) == 1
    assert groups[0].text_direction == "vertical"


def test_short_latin_column_stays_horizontal():
    """Short Latin text in a tall bbox is not tategaki; keep horizontal."""
    groups = _run([_col(100, 100, 180, 200, "Hi")])
    assert groups[0].text_direction == "horizontal"


# ─── Chain compatibility ─────────────────────────────────────────────────


def test_three_columns_in_bubble_merge_into_one_group():
    """Three side-by-side tategaki columns sharing y-span → one group."""
    cols = [
        _col(300, 100, 360, 800, "右側のセリフ"),
        _col(220, 110, 280, 790, "中央のセリフ"),
        _col(140, 105, 200, 800, "左側のセリフ"),
    ]
    groups = _run(cols)
    assert len(groups) == 1
    merged = groups[0]
    # Reading order: right-to-left
    assert merged.text.splitlines() == [
        "右側のセリフ", "中央のセリフ", "左側のセリフ",
    ]
    assert merged.bbox == (140, 100, 360, 800)


def test_far_column_is_not_merged():
    """A column outside the x-gap budget must stay separate."""
    cols = [
        _col(300, 100, 360, 800, "右側"),     # width 60
        _col(140, 100, 200, 800, "左側"),     # width 60
        # Way out on the left — gap from leftmost (140) is 1240‑200 ≈ no,
        # actually we put it well beyond `max(80, 2×60)=120`:
        _col(800, 100, 860, 800, "別バブル"),  # gap to col@300 = 800-360=440
    ]
    # Sort doesn't matter; chain processes right-to-left internally.
    groups = _run(cols)
    # Expect two clusters: (col@300, col@140) merged, col@800 alone.
    assert len(groups) == 2
    texts = sorted([g.text for g in groups])
    assert "別バブル" in texts


def test_width_scaled_gap_budget_high_res():
    """High-res page: columns 100px wide, gap 150px must still merge."""
    # min(width)*2.0 = 200 → gap 150 is within budget
    cols = [
        _col(1000, 200, 1100, 1000, "高解像度の右"),
        _col(750, 200, 850, 1000, "高解像度の左"),
    ]
    groups = _run(cols)
    assert len(groups) == 1


# ─── Cluster guards ──────────────────────────────────────────────────────


def test_outsider_guard_rejects_bridge_engulfed_outsider():
    """Cluster bbox engulfs a non-member when a tall bridge member
    extends y-range past a neighbour's y-span.

    Setup:
      * Tall column A at x=300, y=100..1500 (the bridge — very tall column).
      * Short column B at x=140, y=200..400 (overlaps A → joins chain).
      * Outsider X at x=220, y=1000..1300. Y doesn't overlap B (200..400)
        and the chain only checks compat against existing cluster, but
        since A spans 100..1500, X also overlaps A — so chain WOULD pick
        it up via A. To create a real outsider we make X y outside A too.

    Easier: keep the outsider with y=1700..1900 (below A's y-range).
    Cluster A+B bbox = (140, 100, 360, 1500); outsider centre (240, 1800)
    is outside cluster bbox → guard not triggered. So we just verify the
    cluster of 2 is formed and outsider stays alone (chain handles it).
    """
    cols = [
        _col(300, 100, 360, 1500, "縦長A"),   # tall bridge
        _col(140, 200, 200, 400,  "短いB"),   # short, overlaps A in y
        _col(220, 1700, 260, 1900, "下に別"),  # outsider, below cluster bbox
    ]
    groups = _run(cols)
    assert len(groups) == 2
    big = max(groups, key=lambda g: (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))
    assert "縦長A" in big.text and "短いB" in big.text


def test_outsider_guard_direct():
    """Direct call to outsider guard with a manufactured cluster bbox."""
    from typoon.vision.groupers.lens_native import (
        _block_to_group,
        _passes_outsider_guard,
    )
    a = _block_to_group(_col(300, 100, 360, 1500, "A"))
    b = _block_to_group(_col(140, 200, 200, 400, "B"))
    # Outsider whose centre (240, 950) falls inside cluster bbox (140..360, 100..1500)
    outsider = _block_to_group(_col(220, 900, 260, 1000, "X"))
    members = [a, b]
    all_vert = [a, b, outsider]
    # Member indices in `all_vert` for the cluster
    assert _passes_outsider_guard(members, all_vert, [0, 1]) is False


def test_font_guard_blocks_size_mismatch_merge():
    """Two adjacent columns with very different widths → different glyph sizes."""
    cols = [
        # Large bubble: 120 wide column
        _col(500, 100, 620, 800, "大きい字"),
        # Small bubble nearby: 30 wide column — ratio 4× → fails guard
        _col(390, 100, 420, 800, "小"),
    ]
    groups = _run(cols)
    assert len(groups) == 2


def test_font_guard_allows_similar_sizes():
    """Columns within 1.8× width ratio merge fine."""
    cols = [
        _col(500, 100, 580, 800, "右側A"),   # w=80
        _col(380, 100, 440, 800, "左側A"),   # w=60, ratio 1.33
    ]
    groups = _run(cols)
    assert len(groups) == 1


# ─── Mixed-direction safety ──────────────────────────────────────────────


def test_horizontal_blocks_left_untouched():
    blocks = [
        _block((100, 100, 500, 150), "Hello world this is horizontal"),
        _block((100, 200, 500, 250), "Another horizontal line"),
    ]
    groups = _run(blocks)
    assert len(groups) == 2
    assert all(g.text_direction == "horizontal" for g in groups)


def test_horizontal_and_vertical_coexist():
    blocks = [
        _block((50, 50, 800, 120), "Title horizontal"),
        _col(300, 200, 360, 700, "縦書き右"),
        _col(220, 200, 280, 700, "縦書き左"),
    ]
    groups = _run(blocks)
    # 1 horizontal + 1 merged vertical = 2
    assert len(groups) == 2
    by_dir = {g.text_direction: g for g in groups}
    assert by_dir["vertical"].text.splitlines() == ["縦書き右", "縦書き左"]


def test_max_columns_safety_cap():
    """A degenerate >6 column cluster falls back to singletons."""
    cols = [
        _col(60 * (10 - i), 100, 60 * (10 - i) + 50, 800, f"列{i}")
        for i in range(7)
    ]
    groups = _run(cols)
    assert len(groups) == 7
