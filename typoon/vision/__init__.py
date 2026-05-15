"""Vision pipeline — async-first stage protocols + composable runtime.

Public surface:
  - VisionPipelineSpec   — declarative composition + named PRESETS
  - VisionRuntime        — assembled stages with concurrency gates
  - build_vision_runtime — spec → runtime factory
  - contracts            — Protocols + frozen records (TextBlock, BubbleGroup, ...)

Backends live under detectors/, groupers/, recognizers/, erasers/.
Add a new backend by:
  1. Implement the relevant Protocol from `contracts`
  2. Add a Literal value to the matching Id in `pipeline`
  3. Add a `match` arm in `runtime._build_*`
"""

from .contracts import (
    BubbleGroup,
    DetectionResult,
    TextBlock,
    TextDetector,
    TextEraser,
    TextGrouper,
    TextMask,
    TextRecognizer,
)
from .pipeline import (
    DETECTORS_SHIPPING_TEXT,
    GROUPERS_REQUIRING_LENS,
    PRESETS,
    DetectorId,
    EraserId,
    GrouperId,
    RecognizerId,
    VisionPipelineSpec,
)
from .runtime import VisionRuntime, build_vision_runtime


__all__ = [
    "VisionPipelineSpec",
    "PRESETS",
    "DETECTORS_SHIPPING_TEXT",
    "GROUPERS_REQUIRING_LENS",
    "DetectorId",
    "GrouperId",
    "RecognizerId",
    "EraserId",
    "VisionRuntime",
    "build_vision_runtime",
    "TextDetector",
    "TextGrouper",
    "TextRecognizer",
    "TextEraser",
    "TextBlock",
    "DetectionResult",
    "BubbleGroup",
    "TextMask",
]
