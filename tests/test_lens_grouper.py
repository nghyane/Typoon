"""Lens-native grouper — block → BubbleGroup mapping tests.

Synthetic input only; no network, no model.
"""

from __future__ import annotations

import asyncio

import numpy as np

from typoon.vision.contracts import DetectionResult, TextBlock
from typoon.vision.groupers.lens_native import LensNativeGrouper


def _block(bbox, text):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
    )


def test_one_block_yields_one_group():
    grouper = LensNativeGrouper()
    detection = DetectionResult(
        blocks=(_block((10, 10, 110, 60), "Hello world"),),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    groups = asyncio.run(grouper.group(image, detection, "en"))
    assert len(groups) == 1
    g = groups[0]
    assert g.text == "Hello world"
    # bbox is the Lens block (no word geometry → falls back to block bbox)
    # expanded by the minimum container padding (4 px) and clipped to page.
    # +1 on the high edge: container polygon AABB rounds up at the
    # bottom-right so the rect still encloses the original polygon.
    assert g.bbox == (6, 6, 115, 65)
    assert g.source == "lens"
    assert len(g.text_masks) == 1
    assert len(g.erase_masks) == 1


def test_long_text_classified_as_narration():
    """Long captions (>30 chars) → narration shape_kind=dialogue.

    The old `uppercase_heavy → burst` heuristic was wrong: long
    uppercase narration is still narration, not SFX. SFX is now
    detected from geometry (rotation, aspect, char count).
    """
    grouper = LensNativeGrouper()
    text = "DAMMIT, WHY CAN'T I REMEMBER ANY OF IT?!"
    detection = DetectionResult(
        blocks=(_block((10, 10, 200, 110), text),),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    groups = asyncio.run(grouper.group(image, detection, "en"))
    # Narration profile maps to shape_kind=dialogue (plain stroke, not glow)
    assert groups[0].shape_kind == "dialogue"


def test_rotated_short_text_classified_as_burst():
    """Rotated short text → SFX → shape_kind=burst (glow halo)."""
    grouper = LensNativeGrouper()
    detection = DetectionResult(
        blocks=(
            TextBlock(
                bbox=(10, 10, 200, 60),
                polygon=None,
                confidence=1.0,
                text="BLUSH",
                detector="lens_blocks",
                rotation_deg=-15.0,
            ),
        ),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    groups = asyncio.run(grouper.group(image, detection, "en"))
    assert groups[0].shape_kind == "burst"


def test_lowercase_classified_as_dialogue():
    grouper = LensNativeGrouper()
    detection = DetectionResult(
        blocks=(_block((10, 10, 200, 110), "this is a normal dialogue line"),),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    groups = asyncio.run(grouper.group(image, detection, "en"))
    assert groups[0].shape_kind == "dialogue"


def test_erase_mask_matches_block_extent():
    """Erase mask follows the tight Lens word_union polygon (no per-block
    dilation). When the block has no word geometry the polygon collapses
    to the block bbox itself, expanded by the minimum mask padding.
    """
    grouper = LensNativeGrouper()
    detection = DetectionResult(
        blocks=(_block((50, 50, 150, 150), "test"),),
        text_already_recognized=True,
        page_size=(500, 500),
    )
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    groups = asyncio.run(grouper.group(image, detection, "en"))
    em = groups[0].erase_masks[0]
    # No words / no line geometry → word_union fallback = block bbox 100×100,
    # dilated by `_MASK_PAD_MIN_PX` (= 2 px) — bumped from 1 so Lens
    # under-coverage on diagonal strokes / ascenders no longer ghosts.
    assert (em.x, em.y) == (48, 48)
    assert em.image.shape == (104, 104)
    assert (em.image == 255).all()
