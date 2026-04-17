"""PP-OCR text detection — auto-selects best backend.

CoreML (Mac, thread-safe, ~58ms) > MLX (Mac ARM, faster but single-stream) > ONNX.
"""

import sys
from pathlib import Path


def _coreml_available(models_dir: str) -> bool:
    if sys.platform != "darwin":
        return False
    if not Path(models_dir).joinpath("ppocr-det.mlpackage").exists():
        return False
    try:
        import coremltools  # noqa: F401
        return True
    except ImportError:
        return False


class TextDetector:
    """PP-OCR det with auto backend selection."""

    def __init__(self, model_path: str, config_path: str) -> None:
        models_dir = str(Path(model_path).parent)
        if _coreml_available(models_dir):
            from .coreml import TextDetector as _Impl
            self._impl = _Impl(str(Path(models_dir) / "ppocr-det.mlpackage"))
        elif sys.platform == "darwin":
            from .mlx import TextDetector as _Impl
            self._impl = _Impl(model_path, config_path)
        else:
            from .onnx import TextDetector as _Impl
            self._impl = _Impl(model_path, config_path)

    def detect(self, image):
        return self._impl.detect(image)
