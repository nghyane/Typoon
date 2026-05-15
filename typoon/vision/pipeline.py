"""VisionPipelineSpec — declarative composition with named presets.

Frozen, slotted, validated at construction. Discriminated unions via Literal
let the type-checker enumerate every backend.
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
    "GROUPERS_REQUIRING_LENS",
    "VisionPipelineSpec",
    "PRESETS",
]


DetectorId   = Literal["lens_blocks", "ppocr_dbnet", "bing_blocks"]
GrouperId    = Literal["lens_native", "ppocr_yolo_union_find"]
RecognizerId = Literal["none", "manga_ocr", "apple_vision",
                       "windows_ocr", "tesseract"]
EraserId     = Literal["aot_gan", "median_only"]


# Capability sets — single source of truth for cross-stage validation.
# A detector "ships text" if it returns recognised text inside TextBlock,
# making the recognizer stage redundant (recognizer="none" is valid).
DETECTORS_SHIPPING_TEXT: frozenset[DetectorId] = frozenset({
    "lens_blocks",
    "bing_blocks",
})

# A grouper that exclusively consumes one detector's bubble-shaped output
# (vs line-shaped output that needs YOLO scope merging).
GROUPERS_REQUIRING_LENS: frozenset[GrouperId] = frozenset({"lens_native"})


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
    eraser:     EraserId     = "aot_gan"

    page_concurrency:   int = 4    # how many pages scan in parallel (RAM-bound)
    detect_concurrency: int = 8    # detector calls in flight (HTTP / GPU)
    erase_concurrency:  int = 2    # eraser calls in flight (GPU contention)

    def __post_init__(self) -> None:
        if (
            self.grouper in GROUPERS_REQUIRING_LENS
            and self.detector != "lens_blocks"
        ):
            raise ValueError(
                f"grouper={self.grouper!r} only consumes lens_blocks output; "
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
    "bing": VisionPipelineSpec(
        detector="bing_blocks",
        grouper="ppocr_yolo_union_find",
        recognizer="none",
        # Bing has stricter rate limits than Lens — keep concurrency low
        page_concurrency=2,
        detect_concurrency=2,
    ),
    "offline": VisionPipelineSpec(
        detector="ppocr_dbnet",
        grouper="ppocr_yolo_union_find",
        recognizer="apple_vision",
        page_concurrency=2,
        detect_concurrency=2,
    ),
    "manga_ja": VisionPipelineSpec(
        detector="ppocr_dbnet",
        grouper="ppocr_yolo_union_find",
        recognizer="manga_ocr",
        page_concurrency=2,
    ),
}
