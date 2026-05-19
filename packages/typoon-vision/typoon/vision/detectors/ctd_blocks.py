"""Comic Text Detector — async TextDetector adapter (ONNX/CoreML backend).

Wraps CTDBackend (ONNX inference) in the VisionPipeline contract.
CoreML is used automatically on macOS Apple Silicon via onnxruntime.

Emits TextBlock with:
  - DBNet tight polygon  (minAreaRect from fused text pixel mask)
  - UNet pixel text_mask (per-block crop of the fused mask)
  - text_direction       (inferred from bbox aspect + lang hint)

DetectionResult.bubble_mask carries the full-page UNet segmentation mask
for BubbleIndex safe-area lookup in render stage.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from ..contracts import DetectionResult, TextBlock, TextMask


__all__ = ["CTDDetector"]

logger = logging.getLogger(__name__)


class CTDDetector:
    """Comic Text Detector (ONNX/CoreML).

    The ONNX model is downloaded from mayocream/comic-text-detector-onnx
    on first use via ModelHub.resolve_ctd(). Backend lazy-loaded on
    first detect() call.
    """

    name = "ctd_blocks"

    def __init__(self, onnx_path: Path | str) -> None:
        self._onnx_path = Path(onnx_path)
        self._backend   = None

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        backend = self._get_backend()
        result  = await asyncio.to_thread(backend.detect, image)

        h, w = image.shape[:2]
        blocks: list[TextBlock] = []

        for region in result.text_regions:
            direction = _infer_direction(region.bbox, lang)
            mask = TextMask(
                x=region.mask_x,
                y=region.mask_y,
                image=region.text_mask,
            )
            blocks.append(TextBlock(
                bbox=region.bbox,
                polygon=region.polygon,
                confidence=region.confidence,
                text=None,
                detector=self.name,
                text_mask=mask,
                text_direction=direction,
            ))

        return DetectionResult(
            blocks=tuple(blocks),
            text_already_recognized=False,
            page_size=(w, h),
            bubble_mask=result.bubble_mask,
        )

    def _get_backend(self):
        if self._backend is None:
            from typoon.vision._backends.ctd import CTDBackend
            logger.info("Loading CTD ONNX model (CoreML)...")
            self._backend = CTDBackend(self._onnx_path)
            logger.info("CTD backend ready")
        return self._backend


def _infer_direction(bbox: tuple[int, int, int, int], lang: str | None) -> str:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    is_cjk = lang in ("ja", "zh", "zh-CN", "zh-TW", "ko")
    if is_cjk and bh > bw * 1.5:
        return "vertical"
    return "horizontal"
