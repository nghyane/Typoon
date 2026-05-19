"""Vision pipeline — async-first stage protocols + composable runtime.

Import directly:
    from typoon.vision.contracts import (
        TextBlock, BubbleGroup, DetectionResult, TextMask,
        TextDetector, TextGrouper, TextRecognizer, TextEraser,
    )
    from typoon.vision.pipeline import VisionPipelineSpec, PRESETS
    from typoon.vision.runtime  import VisionRuntime, build_vision_runtime

Backends live under detectors/, groupers/, recognizers/, erasers/.
Add a new backend by:
  1. Implement the relevant Protocol from `contracts`
  2. Add a Literal value to the matching Id in `pipeline`
  3. Add a `match` arm in `runtime._build_*`
"""
