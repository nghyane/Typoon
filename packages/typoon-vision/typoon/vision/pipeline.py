"""VisionPipelineSpec — declarative composition with named presets.

Frozen, slotted, validated at construction. Discriminated unions via Literal
let the type-checker enumerate every backend.

Supported stacks:
  lens      — Lens OCR + Comic-DETR + LensNativeGrouper   (primary, online)
  ctd_manga — CTD detector + CTDNativeGrouper + MangaOCR  (offline manga JP)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self


__all__ = [
    "DetectorId",
    "GrouperId",
    "RecognizerId",
    "EraserId",
    "DETECTORS_SHIPPING_TEXT",
    "VisionPipelineSpec",
    "PRESETS",
]


DetectorId   = Literal["lens_blocks", "ctd_blocks"]
GrouperId    = Literal["lens_native", "ctd_native"]  # ppocr_yolo_union_find removed
RecognizerId = Literal["none", "manga_ocr", "apple_vision",
                       "windows_ocr", "tesseract"]
EraserId     = Literal["hybrid"]


# Detectors that ship recognised text inside TextBlock (recognizer=none valid)
DETECTORS_SHIPPING_TEXT: frozenset[DetectorId] = frozenset({"lens_blocks"})

# Grouper → required detector constraint
_GROUPER_REQUIRES: dict[GrouperId, DetectorId] = {
    "lens_native": "lens_blocks",
    "ctd_native":  "ctd_blocks",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionPipelineSpec:
    """Declarative pipeline composition.

    Validates cross-stage compatibility at construction. Concurrency knobs
    are per pipeline so different presets can tune for their own constraints
    (HTTP rate vs GPU contention).
    """
    detector:   DetectorId   = "lens_blocks"
    grouper:    GrouperId    = "lens_native"
    recognizer: RecognizerId = "none"
    eraser:     EraserId     = "hybrid"

    page_concurrency:   int = 4
    detect_concurrency: int = 8
    erase_concurrency:  int = 2

    def __post_init__(self) -> None:
        required_detector = _GROUPER_REQUIRES.get(self.grouper)
        if required_detector and self.detector != required_detector:
            raise ValueError(
                f"grouper={self.grouper!r} requires detector={required_detector!r}; "
                f"got detector={self.detector!r}"
            )
        if (
            self.recognizer == "none"
            and self.detector not in DETECTORS_SHIPPING_TEXT
        ):
            raise ValueError(
                f"recognizer=none requires a detector that ships text "
                f"(one of {sorted(DETECTORS_SHIPPING_TEXT)}); "
                f"got detector={self.detector!r}"
            )
        for field_name in ("page_concurrency", "detect_concurrency", "erase_concurrency"):
            value = getattr(self, field_name)
            if value < 1:
                raise ValueError(f"{field_name} must be >= 1, got {value}")

    @classmethod
    def preset(cls, name: str) -> Self:
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
        return PRESETS[name]

    def with_overrides(self, **kwargs) -> Self:
        """Return a copy with overrides applied. Validation runs again."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


PRESETS: dict[str, VisionPipelineSpec] = {
    "lens": VisionPipelineSpec(),
    "lens_balanced": VisionPipelineSpec(
        page_concurrency=8,
        detect_concurrency=15,
    ),
    "ctd_manga": VisionPipelineSpec(
        detector="ctd_blocks",
        grouper="ctd_native",
        recognizer="manga_ocr",
        page_concurrency=2,
        detect_concurrency=2,
    ),
}
