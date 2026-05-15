"""Lens block detector — filter rule tests.

Pure-function tests on the filter helpers; no network required.
"""

from __future__ import annotations

from typoon.vision.contracts import TextBlock
from typoon.vision.detectors.lens_blocks import (
    _bbox_too_large_for_text,
    _bbox_too_small,
    _filter_blocks,
    _is_decoration_only,
)


def _block(bbox, text, *, conf: float = 1.0) -> TextBlock:
    return TextBlock(
        bbox=bbox, polygon=None, confidence=conf,
        text=text, detector="lens_blocks",
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
