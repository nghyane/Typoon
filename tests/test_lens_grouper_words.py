"""Lens-native grouper word-union mask + rotation propagation tests."""

from __future__ import annotations

import asyncio

import numpy as np

from typoon.vision.contracts import (
    BubbleGroup,
    DetectionResult,
    TextBlock,
    WordBox,
)
from typoon.vision.groupers.lens_native import LensNativeGrouper


def _block(bbox, text, *, words=(), rotation_deg=0.0):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        words=tuple(words), rotation_deg=rotation_deg,
    )


def _run(detection):
    grouper = LensNativeGrouper()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    return asyncio.run(grouper.group(image, detection, "en"))


def test_word_union_mask_when_words_available():
    block = _block(
        (100, 100, 300, 200), "hello world",
        words=[
            WordBox(bbox=(110, 110, 180, 180), text="hello"),
            WordBox(bbox=(200, 110, 280, 180), text="world"),
        ],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    g = groups[0]
    assert not g.used_fallback
    tm = g.text_masks[0]
    # In block-local coords: word1 x∈[10,80), word2 x∈[100,180); gap is
    # x∈[80, 100). Row y=10 is inside both word y-bands.
    assert tm.image[10, 30] == 255      # inside word1
    assert tm.image[10, 90] == 0        # gap between words
    assert tm.image[10, 140] == 255     # inside word2


def test_word_union_mask_smaller_than_block_rect():
    block = _block(
        (0, 0, 300, 100), "hi there",
        words=[
            WordBox(bbox=(10, 10, 40, 90), text="hi"),
            WordBox(bbox=(60, 10, 250, 90), text="there"),
        ],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    painted = (tm.image == 255).sum()
    total = tm.image.size
    # Painted area should be strictly less than the full block rect
    assert painted < total
    # ... but more than zero (at least the two word rects)
    assert painted > 0


def test_block_rect_fallback_when_words_empty():
    block = _block((10, 10, 110, 60), "fallback")
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    g = groups[0]
    assert g.used_fallback
    assert (g.text_masks[0].image == 255).all()


def test_rotation_propagated_from_block_to_group():
    block = _block((10, 10, 200, 80), "tilted", rotation_deg=12.5)
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    assert groups[0].rotation_deg == 12.5


def test_word_outside_block_bbox_clipped():
    """A word whose bbox sits partly outside the block should not crash
    and should only paint inside the block."""
    block = _block(
        (100, 100, 200, 200), "x",
        words=[WordBox(bbox=(50, 50, 250, 250), text="x")],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    groups = _run(det)
    tm = groups[0].text_masks[0]
    assert tm.image.shape == (100, 100)
    assert (tm.image == 255).all()
