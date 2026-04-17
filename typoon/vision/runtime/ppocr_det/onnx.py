"""PP-OCRv5 mobile text detection — ONNX Runtime backend."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TextDetector:
    """PP-OCR text detection via ONNX Runtime."""

    def __init__(self, model_path: str, config_path: str) -> None:
        self._onnx_path = str(Path(model_path).with_suffix(".onnx"))
        self._session: ort.InferenceSession | None = None

    def _ensure_loaded(self) -> ort.InferenceSession:
        if self._session is None:
            self._session = ort.InferenceSession(
                self._onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("PP-OCR det ready (ONNX, %s)", self._onnx_path)
        return self._session

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run detection. image: RGB uint8 HWC, preprocessed/padded.

        Returns: probability map float32 HW, same size as input.
        """
        session = self._ensure_loaded()

        # HWC uint8 → NCHW float32, ImageNet normalize
        x = image.astype(np.float32) * (1.0 / 255.0)
        x = np.transpose(x, (2, 0, 1))[np.newaxis]
        x = (x - _MEAN) / _STD

        input_name = session.get_inputs()[0].name
        (prob,) = session.run(None, {input_name: x})

        # NCHW [1, 1, H, W] → HW
        return prob[0, 0]
