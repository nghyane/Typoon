"""Shared data types for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class VisualTextGroup:
    """Canonical source of truth for one accepted visual text group.

    No compatibility aliases. Callers use explicit field names:
    - render_polygon for render geometry
    - erase_masks for inpaint masks
    - text_polygon / text_masks for OCR/debug geometry
    """

    text: str
    confidence: float
    text_polygon: list[list[float]]
    render_polygon: list[list[float]]
    text_bbox: list[int]
    mask_bbox: list[int]
    fit_bbox: list[int]
    erase_bbox: list[int]
    scope_bbox: list[int] | None = None
    scope_confidence: float | None = None
    text_masks: list[TextMask] = field(default_factory=list)
    erase_masks: list[TextMask] = field(default_factory=list)
    source: str = "unknown"
    unit_indices: list[int] = field(default_factory=list)
    accepted: bool = True
    reject_reason: str | None = None
