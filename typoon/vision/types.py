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
    crop: np.ndarray            # RGB uint8 (H, W, 3)
    confidence: float
    mask: TextMask | None


@dataclass
class DetectionOutput:
    """Detection results for one page."""

    regions: list[TextRegion]
    prob_image: np.ndarray | None  # grayscale uint8 (H, W)


@dataclass
class DetectedGroup:
    """One accepted text group — output of detect → group → OCR pipeline."""

    text: str
    confidence: float
    text_polygon: list[list[float]]
    render_polygon: list[list[float]]
    text_box: list[int]
    mask_box: list[int]
    fit_box: list[int]
    erase_box: list[int]
    scope_box: list[int] | None = None
    scope_confidence: float | None = None
    text_masks: list[TextMask] = field(default_factory=list)
    erase_masks: list[TextMask] = field(default_factory=list)
    source: str = "unknown"
    unit_indices: list[int] = field(default_factory=list)
    accepted: bool = True
    reject_reason: str | None = None
    shape_kind: str = "dialogue"   # dialogue | burst




# ── Grouping pipeline internal state ─────────────────────────────────


@dataclass
class UnitState:
    """Mutable state for one detected text unit during grouping."""

    idx: int
    region: TextRegion
    bbox: list[int]
    text: str = ""
    confidence: float = 0.0
    is_noise: bool = False
    noise_reason: str | None = None
    scope_idx: int | None = None


@dataclass
class ScopeState:
    """YOLO bubble scope — scope hint only, not final geometry."""

    idx: int
    bbox: list[int]
    confidence: float


@dataclass
class GroupState:
    """Mutable state for one text group during grouping pipeline."""

    idx: int
    unit_indices: list[int]
    scoped: bool
    scope_idx: int | None
    raw_bbox: list[int]
    ocr_bbox: list[int]
    fit_bbox: list[int]
    text: str = ""
    confidence: float = 0.0
    accepted: bool = False
    reject_reason: str | None = None
    scope_bbox: list[int] | None = None
    median_angle: float = 0.0
    shape_kind: str = "dialogue"   # dialogue | burst


@dataclass
class ScanState:
    """Intermediate state accumulating results through the grouping pipeline."""

    image: np.ndarray
    width: int
    height: int
    units: list[UnitState] = field(default_factory=list)
    scopes: list[ScopeState] = field(default_factory=list)
    groups: list[GroupState] = field(default_factory=list)


