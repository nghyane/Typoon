"""Vision pipeline contracts — Protocols + frozen data records.

Single source of truth for stage interfaces. Detector / Grouper / Recognizer /
Eraser are async Protocols; data passed between stages are frozen, slotted
dataclasses. Stages compose via VisionRuntime; no shared mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


__all__ = [
    "TextMask",
    "WordBox",
    "LineBox",
    "TextBlock",
    "DetectionResult",
    "TypesettingHint",
    "BubbleGroup",
    "TextDetector",
    "TextGrouper",
    "TextRecognizer",
    "TextEraser",
]


# ─── Pixel records ────────────────────────────────────────────────────────


@dataclass(slots=True)
class TextMask:
    """Binary glyph mask in page coordinates. 255 = text, 0 = background.

    Mutable for backward compatibility with packed npz storage. New code
    should treat instances as immutable post-creation.
    """
    x: int
    y: int
    image: np.ndarray  # uint8 (H, W)


@dataclass(frozen=True, slots=True)
class WordBox:
    """Per-word axis-aligned bbox in page pixels.

    Detectors that surface word-level geometry (Lens detailed output)
    populate `TextBlock.words` with these. Groupers use the union to
    build tight erase masks instead of paragraph-rect masks.
    """
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in page pixels
    text: str


@dataclass(frozen=True, slots=True)
class LineBox:
    """Per-line geometry in page pixels.

    Detectors that surface line-level geometry (Lens) populate
    `TextBlock.lines` with these. Used by render for typesetting:
      - line height = font size hint (intrinsic original font px)
      - line count   = aspect-ratio hint (translation should bias toward
        a similar number of lines so rendered bubble fills the same area)
    """
    bbox:         tuple[int, int, int, int]
    text:         str
    rotation_deg: float = 0.0


# ─── Detection contract ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One detected text region. Universal across detector backends."""
    bbox:         tuple[int, int, int, int]              # (x1, y1, x2, y2)
    polygon:      tuple[tuple[float, float], ...] | None # detector-specific shape
    confidence:   float
    text:         str | None    # None unless detector ships recognised text
    detector:     str           # provenance tag (e.g. "lens_blocks")
    text_mask:    TextMask | None = None  # detector may emit per-block glyph mask
    rotation_deg: float = 0.0   # block-level rotation; 0 = axis-aligned horizontal
    words:        tuple[WordBox, ...] = ()  # per-word geometry when available
    lines:        tuple[LineBox, ...] = ()  # per-line geometry when available
    text_direction: str = "horizontal"      # "vertical" | "horizontal" — source script


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Full-page detection output."""
    blocks:                  tuple[TextBlock, ...]
    text_already_recognized: bool                          # bypass recognizer if True
    page_size:               tuple[int, int]               # (width, height)
    rejected:                tuple[tuple[TextBlock, str], ...] = ()  # (block, reason)
    detected_lang:           str | None = None             # ISO 639-1 from detector, if surfaced
    bubble_mask:             "np.ndarray | None" = None    # uint8 (H,W) bubble segmentation
    # Optional bubble-anchor regions from a side-detector (e.g. comic_detr).
    # Each entry: (class_name, (x1, y1, x2, y2), confidence). Class is
    # one of "bubble" | "text_bubble" | "text_free". When present, the
    # grouper uses these as spatial anchors to merge TextBlocks into
    # BubbleGroups (replacing geometry-only heuristics).
    bubble_regions:          tuple[tuple[str, tuple[int, int, int, int], float], ...] = ()


# ─── Grouping contract ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TypesettingHint:
    """Original-text typesetting metadata, passed to render for fit tuning.

    All fields are derived from the detector's per-line geometry. Render
    uses them as priors for the binary-search fit so translated text
    visually matches the original layout (font scale, line count, density).
    """
    font_size_px:     int    # median original line height in page pixels
    line_count:       int    # number of source lines in the bubble
    avg_chars_per_line: float  # density signal — informs balanced wrap target


@dataclass(frozen=True, slots=True)
class BubbleGroup:
    """One bubble emitted by a grouper. Universal across grouping strategies."""
    bbox:           tuple[int, int, int, int]
    polygon:        tuple[tuple[float, float], ...]
    text:           str
    confidence:     float
    text_masks:     tuple[TextMask, ...]
    erase_masks:    tuple[TextMask, ...]
    source:         str                  # "lens" | "ppocr_yolo" | "ctd" | etc.
    shape_kind:     str = "dialogue"     # "dialogue" | "burst"
    used_fallback:  bool = False         # erase mask = full bbox rectangle
    rotation_deg:   float = 0.0          # block-level rotation, propagated to render
    typesetting:    TypesettingHint | None = None  # detector-derived fit hint
    text_direction: str = "horizontal"   # "vertical" | "horizontal" — source script direction


# ─── Stage protocols ──────────────────────────────────────────────────────


@runtime_checkable
class TextDetector(Protocol):
    """Detect text regions on a page.

    Implementations may bundle recognition (Lens) or detection-only
    (PP-OCR DBNet); set DetectionResult.text_already_recognized accordingly.
    """
    name: str

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult: ...


@runtime_checkable
class TextGrouper(Protocol):
    """Combine TextBlocks into BubbleGroups."""
    name: str

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]: ...


@runtime_checkable
class TextRecognizer(Protocol):
    """Optional OCR pass when detector doesn't ship recognised text."""
    name: str

    async def recognize(
        self,
        image: np.ndarray,
        groups: tuple[BubbleGroup, ...],
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]: ...


@runtime_checkable
class TextEraser(Protocol):
    """Erase text from page canvas in-place. Returns the same canvas."""
    name: str

    async def erase(
        self,
        canvas: np.ndarray,
        masks: tuple[TextMask, ...],
    ) -> np.ndarray: ...
