"""PP-OCR text detection — auto-selects best backend.

Priority: CoreML (Mac, ANE) > MLX (Mac ARM) > ONNX (all platforms).
"""

import sys
from pathlib import Path

_USE_COREML = False

if sys.platform == "darwin":
    try:
        import coremltools  # noqa: F401
        _USE_COREML = True
    except ImportError:
        pass

if _USE_COREML:
    from .coreml import TextDetector as _CoreMLDetector

    class TextDetector:
        """Wraps CoreML detector, accepts same (model_path, config_path) signature."""

        def __init__(self, model_path: str, config_path: str) -> None:
            mlpackage = str(Path(model_path).parent / "ppocr-det.mlpackage")
            if Path(mlpackage).exists():
                self._impl = _CoreMLDetector(mlpackage)
            else:
                # Fallback to MLX if mlpackage not found
                from .mlx import TextDetector as _MLX
                self._impl = _MLX(model_path, config_path)

        def detect(self, image):
            return self._impl.detect(image)

elif sys.platform == "darwin":
    from .mlx import TextDetector
else:
    from .onnx import TextDetector

__all__ = ["TextDetector"]
