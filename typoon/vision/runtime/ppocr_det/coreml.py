"""PP-OCR text detection — CoreML backend (Mac, GPU/ANE accelerated).

RangeDim model: accepts any input H×W in [128, 2048].
~15-34ms depending on input size.
"""

from __future__ import annotations

import logging

import coremltools as ct
import numpy as np

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TextDetector:
    """PP-OCR text detection via CoreML native. Accepts any input size."""

    def __init__(self, mlpackage_path: str) -> None:
        self._path = mlpackage_path
        self._model: ct.models.MLModel | None = None

    def _ensure_loaded(self) -> ct.models.MLModel:
        if self._model is None:
            self._model = ct.models.MLModel(self._path, compute_units=ct.ComputeUnit.ALL)
            logger.info("PP-OCR det ready (CoreML, %s)", self._path)
        return self._model

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run detection. image: RGB uint8 HWC (any size).

        Returns: probability map float32 HW, same size as input.
        """
        model = self._ensure_loaded()

        # Normalize: HWC uint8 → NCHW float32
        x = image.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        x = (x - _MEAN) / _STD

        out = model.predict({"input": x})
        return next(iter(out.values()))[0, 0]
