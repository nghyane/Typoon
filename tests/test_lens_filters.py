"""Lens detector filter + geometry projection tests.

Pure-function tests on the two helper modules — no network required.
"""

from __future__ import annotations

from typoon.vision.contracts import LineBox, TextBlock
from typoon.vision.detectors.lens.filters import (
    DECORATION_CHARS,
    apply,
    _bbox_too_large_for_text,
    _bbox_too_small,
    _is_decoration_only,
)
from typoon.vision.detectors.lens.geometry import Frame, norm_bbox


def _block(bbox, text, *, conf: float = 1.0, lines=()) -> TextBlock:
    return TextBlock(
        bbox=bbox, polygon=None, confidence=conf,
        text=text, detector="lens_blocks", lines=tuple(lines),
    )


# ─── Filter primitives ────────────────────────────────────────────────────


def test_tiny_bbox_rejected():
    assert _bbox_too_small((10, 10, 30, 25)) is True
    assert _bbox_too_small((10, 10, 100, 60)) is False


def test_decoration_only_rejected():
    assert _is_decoration_only("★") is True
    assert _is_decoration_only("☆ ☆ ☆") is True
    assert _is_decoration_only("...") is True
    assert _is_decoration_only("HELLO") is False
    assert _is_decoration_only("どす") is False
    assert _is_decoration_only("中文") is False


def test_decoration_chars_exposed():
    # Sanity: the public constant carries the expected symbols.
    assert "★" in DECORATION_CHARS
    assert "…" in DECORATION_CHARS


def test_huge_bbox_for_short_text_rejected():
    assert _bbox_too_large_for_text((0, 0, 100, 100), "44") is False
    assert _bbox_too_large_for_text((0, 0, 500, 500), "44") is True
    assert _bbox_too_large_for_text((0, 0, 500, 500), "A" * 100) is False


def test_apply_separates_kept_from_rejected():
    blocks = [
        _block((0, 0, 200, 200), "REAL TEXT HERE"),
        _block((0, 0, 10, 10), "x"),
        _block((0, 0, 100, 100), "★ ☆"),
        _block((0, 0, 800, 800), "44"),
    ]
    kept, rejected = apply(blocks)
    assert len(kept) == 1
    assert kept[0].text == "REAL TEXT HERE"
    assert {r for _, r in rejected} == {"tiny_bbox", "decoration_only", "huge_bbox"}


def test_cross_column_artifact_rejected():
    """Tile-boundary phantom whose lines sit inside the real columns."""
    col_a = _block((1681, 400, 1712, 730), "応,就带着项链来见我")
    col_b = _block((1725, 420, 1755, 750), "是定情信物。如果前辈答")
    col_c = _block((1769, 400, 1801, 760), "盒子里是一串珍珠项链,算")
    phantom = _block(
        (1683, 694, 1802, 763),
        ",算 军答 我",
        lines=(
            LineBox(bbox=(1683, 701, 1712, 733), text="我"),
            LineBox(bbox=(1728, 694, 1756, 747), text="军答"),
            LineBox(bbox=(1773, 708, 1802, 764), text=",算"),
        ),
    )
    kept, rejected = apply([col_a, col_b, col_c, phantom])
    assert phantom not in kept
    assert {r for _, r in rejected} == {"cross_column"}
    assert {b.text for b in kept} == {col_a.text, col_b.text, col_c.text}


def test_cross_column_keeps_normal_multiline_paragraph():
    a = _block(
        (100, 100, 500, 400),
        "line one\nline two\nline three",
        lines=(
            LineBox(bbox=(100, 100, 500, 200), text="line one"),
            LineBox(bbox=(100, 200, 500, 300), text="line two"),
            LineBox(bbox=(100, 300, 500, 400), text="line three"),
        ),
    )
    b = _block((800, 800, 1000, 900), "OTHER")
    kept, rejected = apply([a, b])
    assert len(kept) == 2
    assert not rejected


# ─── geometry.norm_bbox — rotation-aware AABB ─────────────────────────────


def _geom(*, cx_n, cy_n, w_n, h_n, angle_deg=0.0):
    return {
        "center_x": cx_n, "center_y": cy_n,
        "width":    w_n,  "height":   h_n,
        "angle_deg": angle_deg,
    }


def _tile_frame(page_w: int, tile_h: int, origin_y: int = 0) -> Frame:
    return Frame(origin_x=0, origin_y=origin_y,
                 frame_w=page_w, frame_h=tile_h, scale=1)


def test_norm_axis_aligned_passthrough():
    geom = _geom(cx_n=0.5, cy_n=0.5, w_n=0.20, h_n=0.10)
    bbox = norm_bbox(geom, _tile_frame(1000, 1000), (1000, 1000))
    assert bbox == (400, 450, 600, 550)


def test_norm_90deg_rotation_swaps_extents():
    """Watermark rotated 90° → on-page AABB is bh × bw, not bw × bh."""
    geom = _geom(
        cx_n=753.5 / 800, cy_n=1027 / 1136,
        w_n=93 / 800,     h_n=14 / 1136,
        angle_deg=-89.74,
    )
    bbox = norm_bbox(geom, _tile_frame(800, 1136), (800, 1136))
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    assert 12 <= w <= 16
    assert 88 <= h <= 96


def test_norm_45deg_rotation():
    geom = _geom(
        cx_n=0.5, cy_n=0.5,
        w_n=100 / 1000, h_n=20 / 1000,
        angle_deg=45.0,
    )
    bbox = norm_bbox(geom, _tile_frame(1000, 1000), (1000, 1000))
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    assert 82 <= w <= 88
    assert 82 <= h <= 88


def test_norm_origin_y_offset_applied():
    geom = _geom(cx_n=0.5, cy_n=0.5, w_n=0.1, h_n=0.1)
    bbox = norm_bbox(
        geom, _tile_frame(1000, 1000, origin_y=500), (1000, 1500),
    )
    assert bbox is not None
    _, y1, _, y2 = bbox
    assert 949 <= y1 <= 950
    assert 549 <= (y2 - 500) <= 550


def test_norm_bubble_pass_upscale_projection():
    """Bubble pass: crop origin + upscale factor → page coords."""
    # Original bubble at page (200, 300)-(300, 400) — 100×100. Crop pad
    # = 0, upscale ×4 → crop frame 400×400. A glyph centred in frame
    # must land at page centre (250, 350).
    geom = _geom(cx_n=0.5, cy_n=0.5, w_n=0.10, h_n=0.10)
    frame = Frame(origin_x=200, origin_y=300, frame_w=400, frame_h=400, scale=4)
    bbox = norm_bbox(geom, frame, (1000, 1000))
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    # cx/cy normalised → 200, 200 px in crop; /scale = 50, 50 in page;
    # plus origin → (250, 350). w/h normalised → 40 px crop = 10 px page.
    assert (x1, y1, x2, y2) == (245, 345, 255, 355)
