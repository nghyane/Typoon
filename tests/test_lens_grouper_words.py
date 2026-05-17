"""Lens-native grouper mask + rotation propagation tests."""

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


def _block(bbox, text, *, words=(), lines=(), rotation_deg=0.0):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        words=tuple(words),
        lines=tuple(lines) or (LineBox(bbox=bbox, text=text),),
        rotation_deg=rotation_deg,
    )


def _run(detection):
    grouper = LensNativeGrouper()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    return asyncio.run(grouper.group(image, detection, "en"))


def test_mask_is_filled_paragraph_aabb_with_small_dilation():
    """Erase / text masks are a single filled rect per member, sized
    to the paragraph bbox plus a small dilation. The dilation stays
    smaller than the container's word_union padding so the mask never
    leaks past the render polygon."""
    block = _block(
        (100, 100, 300, 200), "hello world",
        words=[
            WordBox(bbox=(110, 110, 180, 180), text="hello"),
            WordBox(bbox=(200, 110, 280, 180), text="world"),
        ],
        lines=[LineBox(bbox=(110, 110, 280, 180), text="hello world")],
    )
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    g = _run(det)[0]

    # text_masks and erase_masks are the same object — one filled
    # rect per member.
    assert g.text_masks is g.erase_masks
    assert len(g.text_masks) == 1
    tm = g.text_masks[0]

    # Filled (no holes between words).
    assert (tm.image == 255).all()
    # Mask sits inside the container polygon — every corner of the
    # mask rect must lie within the bbox axis-aligned bounds.
    mw = tm.image.shape[1]; mh = tm.image.shape[0]
    bx1, by1, bx2, by2 = g.bbox
    assert bx1 <= tm.x and tm.x + mw <= bx2
    assert by1 <= tm.y and tm.y + mh <= by2


def test_mask_uses_block_bbox_when_words_empty():
    """No word geometry → mask still anchored to the paragraph bbox."""
    block = _block((10, 10, 110, 60), "fallback")
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    g = _run(det)[0]
    tm = g.text_masks[0]
    # Covers the original block extent plus the small dilation.
    assert tm.x <= 10 and tm.y <= 10
    assert tm.x + tm.image.shape[1] >= 110
    assert tm.y + tm.image.shape[0] >= 60


def test_rotation_propagated_from_block_to_group():
    block = _block((10, 10, 200, 80), "tilted", rotation_deg=12.5)
    det = DetectionResult(
        blocks=(block,),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    assert _run(det)[0].rotation_deg == 12.5
