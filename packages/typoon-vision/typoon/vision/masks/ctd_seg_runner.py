"""CtdSegRunner — CTD ONNX seg-only inference for bubble mask.

Runs only the seg output (UNet bubble segmentation), skips blk (YOLOv5)
and det (DBNet) decoding. OnnxRuntime still executes the full graph but
only materializes the requested output — ~265ms on CoreML vs ~308ms full.

Used by scan stage for Japanese pages to produce refined bubble_mask that
drives CtdUNetStrategy, giving cleaner erase boundaries than pixel_seg.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from typoon.vision._backends.ctd import _preprocess, _HOLE_CLOSE_RADIUS, _DILATION_RADIUS


__all__ = ["CtdSegRunner"]


class CtdSegRunner:
    """Lazy-load ONNX session, run seg-only inference per page."""

    def __init__(self, onnx_path: Path | str) -> None:
        self._onnx_path = Path(onnx_path)
        self._sess = None

    def _get_sess(self):
        if self._sess is None:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                str(self._onnx_path),
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
        return self._sess

    def run(self, image: np.ndarray) -> np.ndarray:
        """image: RGB uint8 (H,W,3) → refined bubble_mask uint8 (H,W)."""
        orig_h, orig_w = image.shape[:2]
        inp, rw, rh = _preprocess(image)
        [seg_out] = self._get_sess().run(["seg"], {"images": inp})
        seg = seg_out[0, 0]   # (1024, 1024) float

        crop = seg[:rh, :rw]
        full = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (full > 0.5).astype(np.uint8) * 255

        k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (_HOLE_CLOSE_RADIUS * 2 + 1,) * 2)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (_DILATION_RADIUS   * 2 + 1,) * 2)
        closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        return cv2.dilate(closed, k_dilate)
