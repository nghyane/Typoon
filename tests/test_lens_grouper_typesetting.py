"""Lens-native grouper typesetting hint propagation tests."""

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


def _block(bbox, text, *, lines=(), words=(), rotation_deg=0.0):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        words=tuple(words), lines=tuple(lines), rotation_deg=rotation_deg,
    )


def _run(detection):
    grouper = LensNativeGrouper()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    return asyncio.run(grouper.group(image, detection, "en"))


def test_typesetting_hint_built_from_lines():
    block = _block(
        (100, 100, 300, 250), "hello world how are you",
        lines=[
            LineBox(bbox=(100, 100, 300, 130), text="hello world"),
            LineBox(bbox=(100, 135, 300, 165), text="how are"),
            LineBox(bbox=(100, 170, 300, 200), text="you"),
        ],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    g = groups[0]
    assert g.typesetting is not None
    # Median of [30, 30, 30] = 30
    assert g.typesetting.font_size_px == 30
    assert g.typesetting.line_count == 3
    # chars per line: ('hello world' 10 + 'how are' 6 + 'you' 3) / 3
    assert abs(g.typesetting.avg_chars_per_line - (10 + 6 + 3) / 3) < 0.01


def test_typesetting_hint_uses_median_line_height():
    """Outlier line (e.g. tall punctuation tail) should not skew the
    intrinsic font size — median picks the mode."""
    block = _block(
        (0, 0, 200, 200), "a b c",
        lines=[
            LineBox(bbox=(0, 0,   200, 24),  text="a"),  # h=24
            LineBox(bbox=(0, 30,  200, 54),  text="b"),  # h=24
            LineBox(bbox=(0, 60,  200, 144), text="c"),  # outlier h=84
        ],
    )
    det = DetectionResult(
        blocks=(block,), text_already_recognized=True, page_size=(500, 500),
    )
    g = _run(det)[0]
    assert g.typesetting is not None
    assert g.typesetting.font_size_px == 24  # median, not mean


def test_typesetting_hint_none_when_lines_empty():
    """Detectors without per-line geometry (PP-OCR DBNet path) emit
    blocks with `lines=()`; hint must be None so render falls back."""
    block = _block((0, 0, 100, 50), "hello")
    det = DetectionResult(
        blocks=(block,), text_already_recognized=False, page_size=(500, 500),
    )
    g = _run(det)[0]
    assert g.typesetting is None


def test_typesetting_hint_drops_lines_without_text():
    """Lines with empty text contribute neither to height median nor
    char count."""
    block = _block(
        (0, 0, 100, 100), "hi",
        lines=[
            LineBox(bbox=(0, 0,  100, 20), text="hi"),
            LineBox(bbox=(0, 25, 100, 45), text=""),  # ignored upstream by Lens parser
        ],
    )
    det = DetectionResult(
        blocks=(block,), text_already_recognized=True, page_size=(500, 500),
    )
    g = _run(det)[0]
    assert g.typesetting is not None
    assert g.typesetting.line_count == 2  # parser is upstream; here we get what we get
