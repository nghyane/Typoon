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
            out = self._model.generate(
                pv,
                max_new_tokens=64,
                num_beams=1,                       # greedy → scores align 1:1 with sequences
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs = out.sequences                       # [B, prompt_len + gen_len]
        texts = self._tok.batch_decode(seqs, skip_special_tokens=True)

        # Real per-sequence confidence — mean prob of greedy-chosen tokens.
        # The previous hardcoded 1.0 disabled every confidence-gated noise
        # filter for ja, letting "w." and large unscoped fragments through.
        gen_len    = len(out.scores)
        gen_tokens = seqs[:, -gen_len:]                                # [B, gen_len]
        # scores[t]: [B, vocab] logits at step t. Stack → [B, gen_len, vocab].
        stacked    = torch.stack(out.scores, dim=1)
        probs      = torch.softmax(stacked, dim=-1)
        chosen     = probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [B, gen_len]

        eos_id = self._tok.eos_token_id
        pad_id = self._tok.pad_token_id
        # Treat tokens up to (and including) the first EOS as content; mask the rest.
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
