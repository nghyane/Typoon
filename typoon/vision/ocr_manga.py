"""Manga OCR backend — kha-white/manga-ocr-base for Japanese manga.

Uses HuggingFace transformers (PyTorch). Model loads lazily on first
call so non-Japanese projects pay no startup cost.

Apple Vision returns empty / garbage on vertical Japanese in manga
bubbles; manga-ocr is trained on manga109s and handles vertical text,
furigana, manga fonts, and SFX correctly.

Batched dispatch: processor resizes every crop to a fixed 224×224, so
a list-of-crops can be encoded + decoded in a single forward pass with
identical output to sequential calls (verified). On Mac MPS, batched
~1.7× faster than sequential.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

_MODEL_NAME = "kha-white/manga-ocr-base"


def _manga_ocr_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import fugashi  # noqa: F401
        return True
    except ImportError:
        return False


class MangaOcrBackend:
    """Lazy-loaded manga-ocr recognizer.

    Holds tokenizer/processor/model after first `recognize` call. One
    instance per VisionRuntime; shared across pages and chapters.
    """

    # Trained on natural grayscale manga — adaptive-threshold preprocessing
    # in groups.py degrades accuracy. Pipeline checks this flag to skip
    # binarization for ja crops.
    wants_raw = True

    def __init__(self) -> None:
        self._tok: Any = None
        self._proc: Any = None
        self._model: Any = None
        self._device: str = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer,
        )

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._tok   = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._proc  = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        self._model = (
            AutoModelForImageTextToText
            .from_pretrained(_MODEL_NAME)
            .to(self._device)
            .eval()
        )

    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,  # accepted for protocol compat, ignored
    ) -> list[tuple[str, float]]:
        if not crops:
            return []

        self._ensure_loaded()
        import torch

        # Skip degenerate crops; keep slot to preserve order.
        keep_idx: list[int] = []
        pil_imgs: list[Image.Image] = []
        for i, c in enumerate(crops):
            ch, cw = c.shape[:2]
            if ch < 5 or cw < 5:
                continue
            keep_idx.append(i)
            pil_imgs.append(Image.fromarray(c).convert("RGB"))

        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        if not pil_imgs:
            return results

        pv = self._proc(pil_imgs, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            ids = self._model.generate(pv, max_new_tokens=64)
        texts = self._tok.batch_decode(ids, skip_special_tokens=True)

        for slot, raw in zip(keep_idx, texts):
            text = raw.replace(" ", "").strip()
            results[slot] = (text, 1.0 if text else 0.0)
        return results
