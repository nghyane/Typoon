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


# ── Page scan state types ─────────────────────────────────────────────


@dataclass
class TextUnit:
    idx: int
    region: TextRegion
    bbox: list[int]
    unit_ocr_text: str = ""
    unit_ocr_conf: float = 0.0
    is_noise: bool = False
    noise_reason: str | None = None
    scope_idx: int | None = None


@dataclass
class Scope:
    idx: int
    bbox: list[int]
    confidence: float


@dataclass
class TextGroup:
    idx: int
    unit_indices: list[int]
    scoped: bool
    scope_idx: int | None
    raw_bbox: list[int]
    ocr_bbox: list[int]
    fit_bbox: list[int]
    ocr_text: str = ""
    ocr_conf: float = 0.0
    accepted: bool = False
    reject_reason: str | None = None
    scope_bbox: list[int] | None = None
    median_angle: float = 0.0


@dataclass
class PageScanState:
    """Intermediate state for one page through the full scan pipeline."""

    image: np.ndarray
    width: int
    height: int
    units: list[TextUnit] = field(default_factory=list)
    scopes: list[Scope] = field(default_factory=list)
    groups: list[TextGroup] = field(default_factory=list)
