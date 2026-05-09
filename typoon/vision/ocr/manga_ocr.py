"""manga-ocr backend — Japanese vertical text via kha-white/manga-ocr-base.

Apple Vision and Lens both fail on vertical Japanese in stylised manga
bubbles; manga-ocr is trained on manga109s and handles vertical text,
furigana, manga fonts, and SFX correctly. Implemented as `CropOcr` (one
forward pass per crop) because the model takes a single 224×224 image
and returns one string with no geometry.

Lazy load: tokenizer / processor / model are only loaded on the first
`ocr_crops` call, so non-Japanese projects pay no startup cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


_MODEL_NAME = "kha-white/manga-ocr-base"


def is_available() -> bool:
    try:
        import torch         # noqa: F401
        import transformers  # noqa: F401
        import fugashi       # noqa: F401
        return True
    except ImportError:
        return False


class MangaOcrCropOcr:
    """Lazy-loaded manga-ocr recognizer. Batched dispatch via HF processor."""

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

    def ocr_crops(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,  # accepted for protocol parity, ignored
    ) -> list[tuple[str, float]]:
        if not crops:
            return []

        self._ensure_loaded()
        import torch

        # Skip degenerate crops; reserve their slot to preserve order.
        keep_idx: list[int] = []
        pil_imgs: list[Image.Image] = []
        for i, c in enumerate(crops):
            if c.shape[0] < 5 or c.shape[1] < 5:
                continue
            keep_idx.append(i)
            pil_imgs.append(Image.fromarray(c).convert("RGB"))

        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        if not pil_imgs:
            return results

        pv = self._proc(pil_imgs, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                pv,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs  = out.sequences
        texts = self._tok.batch_decode(seqs, skip_special_tokens=True)

        # Per-sequence confidence = mean prob of greedy-chosen content tokens.
        gen_len    = len(out.scores)
        gen_tokens = seqs[:, -gen_len:]
        stacked    = torch.stack(out.scores, dim=1)
        probs      = torch.softmax(stacked, dim=-1)
        chosen     = probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

        eos_id = self._tok.eos_token_id
        pad_id = self._tok.pad_token_id
        valid = torch.ones_like(gen_tokens, dtype=torch.bool)
        if eos_id is not None:
            after_eos = (gen_tokens == eos_id).cumsum(dim=1) > 1
            valid &= ~after_eos
        if pad_id is not None:
            valid &= gen_tokens != pad_id

        masked      = chosen.masked_fill(~valid, 0.0)
        counts      = valid.sum(dim=1).clamp(min=1).to(masked.dtype)
        confidences = (masked.sum(dim=1) / counts).tolist()

        for slot, raw, conf in zip(keep_idx, texts, confidences):
            text = raw.replace(" ", "").strip()
            results[slot] = (text, float(conf) if text else 0.0)
        return results
