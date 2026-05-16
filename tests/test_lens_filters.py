"""Lens block detector — filter rule tests.

Pure-function tests on the filter helpers; no network required.
"""

from __future__ import annotations

from typoon.vision.contracts import LineBox, TextBlock
from typoon.vision.detectors.lens_blocks import (
    _bbox_too_large_for_text,
    _bbox_too_small,
    _filter_blocks,
    _is_decoration_only,
    _norm_geom_to_pixels,
)


def _block(bbox, text, *, conf: float = 1.0, lines=()) -> TextBlock:
    return TextBlock(
        bbox=bbox, polygon=None, confidence=conf,
        text=text, detector="lens_blocks", lines=tuple(lines),
    )


def test_tiny_bbox_rejected():
    assert _bbox_too_small((10, 10, 30, 25)) is True
    assert _bbox_too_small((10, 10, 100, 60)) is False


def test_decoration_only_rejected():
    assert _is_decoration_only("★") is True
    assert _is_decoration_only("☆ ☆ ☆") is True
    assert _is_decoration_only("...") is True
    assert _is_decoration_only("HELLO") is False
    assert _is_decoration_only("どす") is False  # Hiragana category L
    assert _is_decoration_only("中文") is False


def test_huge_bbox_for_short_text_rejected():
    # area=100*100=10000, text='44' chars=2 → 5000/char OK
    assert _bbox_too_large_for_text((0, 0, 100, 100), "44") is False
    # area=500*500=250000, text='44' → 125000/char → reject
    assert _bbox_too_large_for_text((0, 0, 500, 500), "44") is True
    # Long text in big bbox is fine
    assert _bbox_too_large_for_text((0, 0, 500, 500), "A" * 100) is False


def test_filter_blocks_separates_kept_from_rejected():
    blocks = [
        _block((0, 0, 200, 200), "REAL TEXT HERE"),     # OK
        _block((0, 0, 10, 10), "x"),                     # tiny_bbox
        _block((0, 0, 100, 100), "★ ☆"),                 # decoration_only
        _block((0, 0, 800, 800), "44"),                  # huge_bbox
    ]
    kept, rejected = _filter_blocks(blocks)
    assert len(kept) == 1
    assert kept[0].text == "REAL TEXT HERE"
    assert {r for _, r in rejected} == {"tiny_bbox", "decoration_only", "huge_bbox"}


def test_cross_column_artifact_rejected():
    """Lens tile-boundary hallucination: paragraph whose lines sit inside
    other paragraphs (one line per column tail) must be dropped, even
    when no pairwise bbox overlap reaches the dedup IoU threshold.
    """
    # Three tategaki columns (tall, thin, side-by-side)
    col_a = _block((1681, 400, 1712, 730), "応,就带着项链来见我")
    col_b = _block((1725, 420, 1755, 750), "是定情信物。如果前辈答")
    col_c = _block((1769, 400, 1801, 760), "盒子里是一串珍珠项链,算")
    # Phantom "horizontal paragraph" composed of one tail line from each
    # column, all inside the column bboxes above.
    phantom = _block(
        (1683, 694, 1802, 763),
        ",算 军答 我",
        lines=(
            LineBox(bbox=(1683, 701, 1712, 733), text="我"),
            LineBox(bbox=(1728, 694, 1756, 747), text="军答"),
            LineBox(bbox=(1773, 708, 1802, 764), text=",算"),
        ),
    )
    kept, rejected = _filter_blocks([col_a, col_b, col_c, phantom])
    assert phantom not in kept
    assert {r for _, r in rejected} == {"cross_column"}
    assert {b.text for b in kept} == {col_a.text, col_b.text, col_c.text}


def test_cross_column_keeps_normal_paragraph_with_multi_lines():
    """A real multi-line paragraph (lines stacked vertically, not absorbed
    by anyone else) must not be misclassified as cross-column."""
    a = _block(
        (100, 100, 500, 400),
        "line one\nline two\nline three",
        lines=(
            LineBox(bbox=(100, 100, 500, 200), text="line one"),
            LineBox(bbox=(100, 200, 500, 300), text="line two"),
            LineBox(bbox=(100, 300, 500, 400), text="line three"),
        ),
    )
    # An unrelated block far away
    b = _block((800, 800, 1000, 900), "OTHER")
    kept, rejected = _filter_blocks([a, b])
    assert len(kept) == 2
    assert not rejected


# ─── _norm_geom_to_pixels — rotation-aware AABB ────────────────────────────


def _geom(*, cx_n, cy_n, w_n, h_n, angle_deg=0.0):
    """Build a Lens-shaped geometry dict (normalised, plus angle)."""
    return {
        "center_x": cx_n,
        "center_y": cy_n,
        "width": w_n,
        "height": h_n,
        "angle_deg": angle_deg,
    }


def test_norm_geom_axis_aligned_passthrough():
    """No rotation → bbox = centre ± half-extent in page pixels."""
    geom = _geom(cx_n=0.5, cy_n=0.5, w_n=0.20, h_n=0.10)
    bbox = _norm_geom_to_pixels(geom, origin_y=0, page_width=1000, tile_h=1000)
    # cx=500, cy=500, bw=200, bh=100 → (400, 450, 600, 550)
    assert bbox == (400, 450, 600, 550)


def test_norm_geom_90deg_rotation_swaps_extents():
    """Watermark rotated 90° → on-page AABB is bh × bw, not bw × bh.

    Recreates the manhuaren.com watermark from the regression fixture:
    Lens reports `width` along the text reading axis (long: 93px) and
    `height` perpendicular (short: 14px), with angle_deg ≈ -89.74°.
    The page-AABB after rotation must be ~14 wide × ~93 tall.
    """
    # Synthesise: bw_n*page_w = 93, bh_n*page_w = 14 (using tile_h=page_w)
    geom = _geom(
        cx_n=753.5 / 800,
        cy_n=1027 / 1136,
        w_n=93 / 800,
        h_n=14 / 1136,
        angle_deg=-89.74,
    )
    bbox = _norm_geom_to_pixels(geom, origin_y=0, page_width=800, tile_h=1136)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    # Tolerance: integer truncation + tiny rotation gap
    assert 12 <= w <= 16, f"expected width ~14, got {w}"
    assert 88 <= h <= 96, f"expected height ~93, got {h}"


def test_norm_geom_45deg_rotation():
    """45° rotation → AABB is the bounding square of the rotated rect.

    A 100×20 rect rotated 45° has AABB ≈ 85×85 (= (100+20)/√2).
    """
    geom = _geom(
        cx_n=0.5, cy_n=0.5,
        w_n=100 / 1000, h_n=20 / 1000,
        angle_deg=45.0,
    )
    bbox = _norm_geom_to_pixels(geom, origin_y=0, page_width=1000, tile_h=1000)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    assert 82 <= w <= 88
    assert 82 <= h <= 88


def test_norm_geom_origin_y_offset_applied():
    """origin_y from the tile is added to the y component."""
    geom = _geom(cx_n=0.5, cy_n=0.5, w_n=0.1, h_n=0.1)
    bbox = _norm_geom_to_pixels(geom, origin_y=500, page_width=1000, tile_h=1000)
    assert bbox is not None
    _, y1, _, y2 = bbox
    # cy=500, bh=100 → y1 ∈ [origin_y+450, origin_y+449] (truncation)
    assert 949 <= y1 <= 950
    assert 549 <= (y2 - 500) <= 550
