"""Tesseract OCR — page-level via `pytesseract.image_to_data`.

Tesseract supports several layout modes via `--psm`. For comic pages
PSM 11 ("sparse text, find as much as possible, no specific orientation")
gave the best coverage on benchmark pages — it picks up isolated bubble
text that PSM 3 (auto layout) merges into noisy multi-bubble blocks.
Quality is materially below Apple Vision / Google Lens on stylised
manhwa fonts; Tesseract is the cross-platform fallback when neither is
available.

`image_to_data` returns word-level rows with `(block_num, par_num,
line_num)` keys. We collapse rows sharing the same key into one
observation, take the bbox union, and average word confidence into a
[0, 1] score.
"""

from __future__ import annotations

import logging

import numpy as np

from .types import Observation


logger = logging.getLogger(__name__)

_PSM = 11
_MIN_CONF = 0.3

_LANG_MAP: dict[str, str] = {
    "en":    "eng",
    "ja":    "jpn",
    "ko":    "kor",
    "zh":    "chi_sim",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "vi":    "vie",
}


def is_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class TesseractPageOcr:
    def ocr_page(
        self,
        image: np.ndarray,
        *,
        lang: str | None = None,
    ) -> list[Observation]:
        if image.size == 0:
            return []
        import pytesseract
        from pytesseract import Output

        tess_lang = _LANG_MAP.get((lang or "en").lower(), "eng")
        try:
            data = pytesseract.image_to_data(
                image,
                lang=tess_lang,
                config=f"--psm {_PSM}",
                output_type=Output.DICT,
            )
        except Exception as e:
            logger.warning("tesseract OCR failed: %s", e)
            return []

        # Collapse word rows into line-level observations keyed by
        # (block, paragraph, line). Tesseract's confidence is 0-100 with
        # -1 for unknown; clamp those to 0 before averaging.
        lines: dict[tuple[int, int, int], list[dict]] = {}
        for i, raw in enumerate(data["text"]):
            text = raw.strip()
            if not text:
                continue
            try:
                conf = max(0.0, float(data["conf"][i])) / 100.0
            except (TypeError, ValueError):
                conf = 0.0
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append({
                "text": text,
                "conf": conf,
                "x1": int(data["left"][i]),
                "y1": int(data["top"][i]),
                "x2": int(data["left"][i] + data["width"][i]),
                "y2": int(data["top"][i] + data["height"][i]),
            })

        out: list[Observation] = []
        for words in lines.values():
            avg_conf = sum(w["conf"] for w in words) / len(words)
            if avg_conf < _MIN_CONF:
                continue
            x1 = min(w["x1"] for w in words)
            y1 = min(w["y1"] for w in words)
            x2 = max(w["x2"] for w in words)
            y2 = max(w["y2"] for w in words)
            text = " ".join(w["text"] for w in words)
            out.append(Observation(bbox=(x1, y1, x2, y2), text=text, confidence=avg_conf))
        return out
