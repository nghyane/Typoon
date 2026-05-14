"""Pre-OCR shape filter for hatching/screentone artifacts.

PP-OCR detection sometimes fires on dense parallel ink strokes
(screentone, motion lines, hatching) and returns a tight rotated polygon.
The OCR backend then hallucinates plausible text on those fragments
(e.g. "20", "15-2000") with confidence near 1.0, slipping past every
text-quality filter. The structural signal — diagonal narrow stripe —
is reliable and runs before OCR cost.

This test pins the polygon shape thresholds so a regression in the
detector or grouping pipeline cannot quietly re-open the noise path.
"""

from __future__ import annotations

import math

import numpy as np

from typoon.vision.grouping.groups import _is_stripe_cluster, _is_stripe_polygon
from typoon.vision.types import TextRegion, UnitState


def _stripe(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[list[float]]:
    """Rotated rectangle polygon in TL/TR/BR/BL order."""
    ca = math.cos(math.radians(angle_deg))
    sa = math.sin(math.radians(angle_deg))

    def rot(dx: float, dy: float) -> list[float]:
        return [cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]

    return [rot(-w / 2, -h / 2), rot(w / 2, -h / 2), rot(w / 2, h / 2), rot(-w / 2, h / 2)]


def _unit(idx: int, poly: list[list[float]]) -> UnitState:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return UnitState(
        idx=idx,
        region=TextRegion(
            polygon=poly,
            crop=np.zeros((10, 10, 3), dtype=np.uint8),
            confidence=0.7,
            mask=None,
        ),
        bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
    )


def test_axis_aligned_horizontal_text_is_not_stripe():
    poly = [[100, 100], [300, 100], [300, 140], [100, 140]]
    assert not _is_stripe_polygon(poly)


def test_axis_aligned_vertical_japanese_column_is_not_stripe():
    poly = [[100, 100], [130, 100], [130, 400], [100, 400]]
    assert not _is_stripe_polygon(poly)


def test_diagonal_thin_stripe_is_flagged():
    # Hatching fragment, aspect ~5, rotated 30°.
    assert _is_stripe_polygon(_stripe(1580, 1940, 70, 12, 30.0))


def test_steep_diagonal_stripe_is_flagged():
    assert _is_stripe_polygon(_stripe(1580, 1940, 70, 12, 45.0))


def test_mildly_skewed_real_text_is_not_flagged():
    # 8° italic / scan skew, aspect 4 — under tilt threshold.
    assert not _is_stripe_polygon(_stripe(500, 500, 200, 50, 8.0))


def test_tilted_vertical_text_within_tolerance_is_not_flagged():
    # 4° off vertical (90°+4°=94°). axis_deviation = 4°.
    assert not _is_stripe_polygon(_stripe(500, 500, 40, 300, 94.0))


def test_short_aspect_diagonal_polygon_is_not_flagged():
    # Aspect 2 — most glyph clusters have aspect 2-3 even at angles.
    assert not _is_stripe_polygon(_stripe(500, 500, 100, 50, 30.0))


def test_cluster_of_parallel_stripes_is_flagged():
    # Reproduces p3_c5 page 0 group 30: two stripes ~30°/35°.
    units = [
        _unit(0, _stripe(1580, 1940, 70, 12, 30.0)),
        _unit(1, _stripe(1590, 1950, 65, 14, 35.0)),
    ]
    assert _is_stripe_cluster(units, [0, 1])


def test_cluster_with_one_real_glyph_is_kept():
    # Mixed cluster: one stripe + one real horizontal glyph.
    # All-or-nothing: the real glyph rescues the group.
    units = [
        _unit(0, _stripe(1580, 1940, 70, 12, 30.0)),
        _unit(1, [[100, 100], [300, 100], [300, 140], [100, 140]]),
    ]
    assert not _is_stripe_cluster(units, [0, 1])


def test_empty_cluster_is_not_flagged():
    assert not _is_stripe_cluster([], [])
