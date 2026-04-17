"""PP-OCR text detection — CoreML backend (Mac, ANE accelerated).

~15-20ms per page on Apple Silicon via ANE.
EnumeratedShapes: 640×480, 768×544, 960×704, 1280×928.

Handles padding internally to match enumerated shapes.
"""

from __future__ import annotations

import logging

import coremltools as ct
import numpy as np

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Must match shapes used during export
_ENUM_SHAPES = [(640, 480), (768, 544), (960, 704), (1280, 928)]  # (H, W)


def _pick_shape(h: int, w: int) -> tuple[int, int]:
    """Pick smallest enumerated shape that fits h×w."""
    for sh, sw in _ENUM_SHAPES:
        if h <= sh and w <= sw:
            return sh, sw
    return _ENUM_SHAPES[-1]


class TextDetector:
    """PP-OCR text detection via CoreML native."""

    def __init__(self, mlpackage_path: str) -> None:
        self._path = mlpackage_path
        self._model: ct.models.MLModel | None = None

    def _ensure_loaded(self) -> ct.models.MLModel:
        if self._model is None:
            self._model = ct.models.MLModel(self._path, compute_units=ct.ComputeUnit.ALL)
            logger.info("PP-OCR det ready (CoreML, %s)", self._path)
        return self._model

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run detection. image: RGB uint8 HWC (any size, will be padded).

        Returns: probability map float32 HW, same size as input.
        """
        model = self._ensure_loaded()
        h, w = image.shape[:2]

        # Pad to enumerated shape
        th, tw = _pick_shape(h, w)
        padded = np.zeros((th, tw, 3), dtype=np.uint8)
        padded[:h, :w] = image

        # Normalize: HWC uint8 → NCHW float32
        x = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        x = (x - _MEAN) / _STD

        out = model.predict({"input": x})
        prob = next(iter(out.values()))

        # Crop back to input size
        return prob[0, 0, :h, :w]
