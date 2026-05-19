"""Lens-native grouper.

Anchors Lens-emitted TextBlocks into BubbleGroups using
``comic_detr`` region detections (``bubble`` / ``text_bubble`` /
``text_free``). One BubbleGroup per region; un-anchored blocks become
singleton groups so no Lens detection is dropped.

This grouper depends on the detector surfacing ``bubble_regions`` on
the ``DetectionResult``. The Lens preset wires ``comic_detr`` in
``vision.runtime._build_detector`` — running the preset without
comic_detr is a configuration error caught there.

The actual algorithm and mask construction live in two siblings:

  * ``_classify.py``    — SFX / dialogue / narration classifier + profiles
  * ``_spatial_join.py``— region → block assignment, container polygon,
                          and filled-rect erase / text masks
"""

from __future__ import annotations

import asyncio

import numpy as np

from ..contracts import BubbleGroup, DetectionResult
from ._spatial_join import spatial_join


__all__ = ["LensNativeGrouper"]


class LensNativeGrouper:
    """Bubble-anchored grouping for the ``lens`` preset.

    Stateless. The actual algorithm runs on a worker thread because
    mask construction is OpenCV-heavy and blocks the asyncio loop.
    """

    name = "lens_native"

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        return await asyncio.to_thread(
            spatial_join,
            detection.blocks,
            detection.bubble_regions,
            detection.page_size,
        )
