"""AOT-GAN backend — wraps the existing AOTInpainter.

Thin adapter so AOTInpainter conforms to InpaintBackend protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import InpaintBackend

__all__ = ["AOTGANBackend"]


class AOTGANBackend:
    """AOT-GAN inpainting via local ONNX / CoreML model."""

    name = "aot_gan"

    def __init__(self, models_dir: Path | str) -> None:
        self._models_dir = Path(models_dir)
        self._inpainter  = None

    def _get_inpainter(self):
        if self._inpainter is None:
            from typoon.vision._backends.aot import AOTInpainter
            self._inpainter = AOTInpainter(self._models_dir)
        return self._inpainter

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._get_inpainter().inpaint(image_rgb, mask)
