"""Tests for line-to-bubble merging."""

from __future__ import annotations

import math

import numpy as np

from typoon.vision.merge import group_lines
from typoon.vision.types import TextRegion
from .conftest import make_line, _DUMMY_CROP


def _angled_line(x1: float, y1: float, x2: float, y2: float, deg: float) -> TextRegion:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    poly = [[cx + dx * c - dy * s, cy + dx * s + dy * c] for dx, dy in corners]
    return TextRegion(polygon=poly, crop=_DUMMY_CROP, confidence=0.9, mask=None)


# ── Basic merge ──

def test_touching_lines_merge():
    assert len(group_lines([make_line(100, 10, 400, 50), make_line(100, 50, 400, 90)])) == 1


def test_small_gap_merge():
    assert len(group_lines([make_line(100, 10, 400, 50), make_line(100, 55, 400, 95)])) == 1


def test_overlapping_merge():
    assert len(group_lines([make_line(100, 10, 400, 55), make_line(100, 50, 400, 95)])) == 1


def test_paragraph_merge():
    lines = [make_line(100, i * 35, 500, i * 35 + 35) for i in range(4)]
    assert len(group_lines(lines)) == 1


def test_centered_short_line():
    lines = [make_line(100, 10, 400, 50), make_line(200, 55, 300, 95)]
    assert len(group_lines(lines)) == 1


# ── Should separate ──

def test_large_gap_separate():
    assert len(group_lines([make_line(100, 10, 400, 50), make_line(100, 170, 400, 210)])) == 2


def test_no_horizontal_overlap():
    assert len(group_lines([make_line(10, 10, 200, 50), make_line(500, 10, 700, 50)])) == 2


def test_different_angle():
    lines = [make_line(100, 10, 400, 50), make_line(100, 55, 400, 95), _angled_line(100, 110, 400, 150, 15)]
    assert len(group_lines(lines)) == 2


def test_different_height():
    lines = [make_line(100, 10, 400, 50), make_line(100, 55, 400, 95), make_line(100, 200, 400, 300)]
    assert len(group_lines(lines)) == 2


def test_side_by_side():
    lines = [
        make_line(10, 10, 300, 50), make_line(10, 55, 300, 95),
        make_line(450, 10, 750, 50), make_line(450, 55, 750, 95),
    ]
    assert len(group_lines(lines)) == 2


# ── Vertical (Japanese) ──

def _vcol(x1: float, y1: float, x2: float, y2: float) -> TextRegion:
    """Vertical text column: narrow and tall."""
    return make_line(x1, y1, x2, y2)


def test_vertical_side_by_side_merge():
    # Two vertical columns next to each other → one bubble
    lines = [_vcol(100, 10, 130, 200), _vcol(135, 10, 165, 200)]
    assert len(group_lines(lines, vertical=True)) == 1


def test_vertical_large_gap_separate():
    # Two vertical columns far apart → two bubbles
    lines = [_vcol(100, 10, 130, 200), _vcol(400, 10, 430, 200)]
    assert len(group_lines(lines, vertical=True)) == 2


def test_vertical_reading_order_rtl():
    # Japanese reads right-to-left: rightmost column first
    lines = [_vcol(100, 10, 130, 200), _vcol(200, 10, 230, 200), _vcol(300, 10, 330, 200)]
    bubbles = group_lines(lines, vertical=True)
    xs = [max(p[0] for p in b.polygon) for b in bubbles]
    assert xs == sorted(xs, reverse=True)


def test_vertical_paragraph():
    # 4 columns side by side → one bubble
    lines = [_vcol(100 + i * 35, 10, 130 + i * 35, 200) for i in range(4)]
    assert len(group_lines(lines, vertical=True)) == 1


# ── Prob components ──

def test_prob_component_merge():
    prob = np.zeros((200, 500), dtype=np.uint8)
    prob[5:100, 90:410] = 200
    lines = [make_line(100, 10, 400, 50), make_line(100, 55, 400, 95)]
    assert len(group_lines(lines, prob_image=prob)) == 1


# ── BFS connectivity (Gate 5) ──

def _make_manga_page(w: int, h: int, bubble_borders: list[tuple[int, int]]) -> np.ndarray:
    """Simulate manga page: dark artwork background with white bubble interiors.

    bubble_borders: list of (y_start, y_end) for horizontal dark borders.
    """
    # Dark artwork background (like manga panels)
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    # White bubble regions (above and below each border)
    # Fill most of the center as white to simulate bubble interiors
    img[30:h - 30, 80:w - 80] = 255
    # Draw dark borders
    for y_start, y_end in bubble_borders:
        img[y_start:y_end, :] = 0
    return img


def test_bfs_separates_across_border():
    # Two lines in separate bubbles with dark artwork border between them
    img = _make_manga_page(500, 250, bubble_borders=[(110, 130)])
    lines = [make_line(100, 60, 400, 100), make_line(100, 140, 400, 180)]
    # Without image: would merge (small gap, overlapping x)
    assert len(group_lines(lines)) == 1
    # With image: BFS finds dark border → separate
    assert len(group_lines(lines, image=img)) == 2


def test_bfs_allows_same_bubble():
    # Two lines in the same bubble (no border between them)
    img = _make_manga_page(500, 200, bubble_borders=[])
    lines = [make_line(100, 50, 400, 90), make_line(100, 95, 400, 135)]
    assert len(group_lines(lines, image=img)) == 1


# ── Reading order ──

def test_reading_order():
    lines = [make_line(100, 55, 400, 95), make_line(100, 100, 400, 140), make_line(100, 10, 400, 50)]
    bubbles = group_lines(lines)
    assert len(bubbles) == 1
    ys = [min(p[1] for p in l.polygon) for l in bubbles[0].lines]
    assert ys == sorted(ys)
