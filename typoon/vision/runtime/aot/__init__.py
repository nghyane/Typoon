"""AOT inpainting — CoreML native on Mac, ONNX Runtime elsewhere.

Mac Apple Silicon: CoreML .mlpackage with EnumeratedShapes (ANE accelerated)
  128×128: ~4ms, 192×192: ~14ms, 256×256: ~17ms, 384×384: ~37ms

Other platforms: ONNX Runtime with dynamic shapes (CPU/CUDA)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BUCKETS = [128, 192, 256, 384]
_PAD_MOD = 8


def _pick_bucket(h: int, w: int) -> int | None:
    side = max(h, w)
    for b in _BUCKETS:
        if side <= b:
            return b
    return None


# ═══════════════════════════════════════════════════════════════════
# CoreML backend (Mac)
# ═══════════════════════════════════════════════════════════════════


class _CoreMLBackend:
    def __init__(self, mlpackage_path: str) -> None:
        import coremltools as ct
        self._model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
        logger.info("AOT inpaint ready (CoreML, %s)", mlpackage_path)

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image: NCHW float32 [-1,1], mask: NCHW float32 {0,1}. Returns NCHW [-1,1]."""
        out = self._model.predict({"image": image, "mask": mask})
        # coremltools returns dict with output name
        return next(iter(out.values()))


# ═══════════════════════════════════════════════════════════════════
# ONNX backend (Windows/Linux/fallback)
# ═══════════════════════════════════════════════════════════════════


class _OnnxBackend:
    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        logger.info("AOT inpaint ready (ONNX %s, %s)", providers[0], onnx_path)

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {"image": image, "mask": mask})[0]


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


class AOTInpainter:
    """AOT-GAN inpainter. CoreML on Mac, ONNX elsewhere.

    Usage:
        inpainter = AOTInpainter(models_dir)
        result_rgb = inpainter.inpaint(image_rgb, mask_uint8)
    """

    def __init__(self, models_dir: str | Path) -> None:
        self._dir = Path(models_dir)
        self._backend = None
        self._use_coreml = sys.platform == "darwin" and (self._dir / "aot-inpaint.mlpackage").exists()
        if not self._use_coreml and not (self._dir / "aot-inpaint.onnx").exists():
            raise FileNotFoundError(f"Missing aot-inpaint model in {self._dir}")

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self._use_coreml:
            self._backend = _CoreMLBackend(str(self._dir / "aot-inpaint.mlpackage"))
        else:
            self._backend = _OnnxBackend(str(self._dir / "aot-inpaint.onnx"))

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked region.

        Args:
            image_rgb: RGB uint8 (H, W, 3)
            mask: uint8 (H, W), >=127 = inpaint region
        Returns:
            RGB uint8 (H, W, 3) with masked region filled.
        """
        self._ensure_loaded()
        h, w = image_rgb.shape[:2]
        mask_bin = (mask >= 127).astype(np.uint8)

        bucket = _pick_bucket(h, w)
        if bucket is None:
            # Too large — resize down to largest bucket
            scale = _BUCKETS[-1] / max(h, w)
            sh, sw = int(h * scale), int(w * scale)
            result = self._forward_padded(
                cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA),
                cv2.resize(mask_bin, (sw, sh), interpolation=cv2.INTER_NEAREST),
            )
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            result = self._forward_padded(image_rgb, mask_bin)

        mask3 = mask_bin[:, :, None]
        return (result * mask3 + image_rgb * (1 - mask3)).astype(np.uint8)

    def _forward_padded(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Pad to bucket/mod-8, forward, crop back."""
        h, w = image.shape[:2]

        if self._use_coreml:
            bucket = _pick_bucket(h, w) or _BUCKETS[-1]
            ph, pw = bucket, bucket
        else:
            ph = ((h + _PAD_MOD - 1) // _PAD_MOD) * _PAD_MOD
            pw = ((w + _PAD_MOD - 1) // _PAD_MOD) * _PAD_MOD

        padded_img = cv2.copyMakeBorder(image, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
        padded_mask = cv2.copyMakeBorder(mask, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)

        # Normalize: RGB [0,255] → [-1, 1], mask → {0, 1}
        img_t = padded_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 127.5 - 1.0
        mask_t = padded_mask.astype(np.float32)[np.newaxis, np.newaxis]
        img_t = img_t * (1.0 - mask_t)

        out = self._backend.forward(img_t, mask_t)

        # [-1, 1] → [0, 255], crop to original size
        result = ((out[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return result[:h, :w]
