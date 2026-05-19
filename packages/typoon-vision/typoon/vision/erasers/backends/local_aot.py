"""LocalAOTBackend — in-process AOT-GAN inference via the typoon_inpaint
PyO3 extension (Rust/Candle).

For local development where running the inpaint container is overkill.
Production uses the Cloudflare Container path (TyphoonInpaintBackend
+ R2 via spike/inpaint Worker).

Install:
    cd crates/inpaint && maturin develop --release

The extension loads `model.safetensors` once per process and caches the
model in a module-level OnceLock — subsequent calls reuse the warm model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..contracts import InpaintBackend


__all__ = ["LocalAOTBackend"]

logger = logging.getLogger(__name__)


class LocalAOTBackend:
    """In-process AOT-GAN inpaint via the typoon_inpaint PyO3 extension."""

    name = "local_aot"

    def __init__(self, weights_path: Path | str) -> None:
        self._weights_path = str(weights_path)
        # Lazy import — Python process without the extension built shouldn't
        # crash at runtime construction (e.g. test discovery).
        self._module = None

    def _ensure_module(self):
        if self._module is None:
            try:
                import typoon_inpaint as _ti
            except ImportError as e:
                raise RuntimeError(
                    "typoon_inpaint extension not installed. Run: "
                    "cd crates/inpaint && maturin develop --release"
                ) from e
            self._module = _ti
        return self._module

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ti = self._ensure_module()
        return ti.inpaint(image_rgb, mask, self._weights_path)
