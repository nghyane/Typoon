"""Shared data types for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextMask:
    """Binary mask in page coordinates. 255 = text, 0 = background."""

    x: int
    y: int
    image: np.ndarray  # uint8 grayscale (H, W)


@dataclass
class TextRegion:
    """Detected text region with polygon, crop, and optional mask."""

    polygon: list[list[float]]  # [[x, y], ...] corners
    crop: np.ndarray  # RGB uint8 (H, W, 3)
    confidence: float
    mask: TextMask | None


@dataclass
class DetectionOutput:
    """Detection results for one page."""

    regions: list[TextRegion]
    prob_image: np.ndarray | None  # grayscale uint8 (H, W)


@dataclass
class MergedBubble:
    """A bubble with grouped text lines, ready for OCR + concat."""

    polygon: list[list[float]]
    lines: list[TextRegion]
    confidence: float
    masks: list[TextMask]



