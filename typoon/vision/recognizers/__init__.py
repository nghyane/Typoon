"""Recognizer backends — fill BubbleGroup.text when detector doesn't ship it.

Wraps the existing PageOcr / CropOcr backends in vision.ocr in the unified
TextRecognizer Protocol. Selection happens in factories.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ..contracts import BubbleGroup


__all__ = ["PageOcrRecognizer", "MangaOcrRecognizer"]


class PageOcrRecognizer:
    """Adapt a PageOcr backend (apple_vision/windows/tesseract) to TextRecognizer.

    Calls ocr_page once on the full image, then assigns each observation to
    the group whose bbox contains the observation's centre.
    """

    def __init__(self, backend, *, name: str) -> None:
        self._backend = backend
        self.name = name

    async def recognize(
        self,
        image: np.ndarray,
        groups: tuple[BubbleGroup, ...],
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        if not groups:
            return groups
        observations = await asyncio.to_thread(
            self._backend.ocr_page, image, lang=lang,
        )
        return _assign(groups, observations)


class MangaOcrRecognizer:
    """Adapt a CropOcr backend (manga-ocr) to TextRecognizer.

    Crops each group's bbox and runs OCR per crop.
    """

    name = "manga_ocr"

    def __init__(self, backend) -> None:
        self._backend = backend

    async def recognize(
        self,
        image: np.ndarray,
        groups: tuple[BubbleGroup, ...],
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        if not groups:
            return groups
        crops = [_crop(image, g.bbox) for g in groups]
        results = await asyncio.to_thread(self._backend.ocr_crops, crops, lang=lang)
        return tuple(
            _replace(g, text=text.strip(), confidence=float(conf))
            for g, (text, conf) in zip(groups, results)
        )


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(w, bbox[2]), min(h, bbox[3])
    return image[y1:y2, x1:x2]


def _assign(
    groups: tuple[BubbleGroup, ...], observations,
) -> tuple[BubbleGroup, ...]:
    buckets: dict[int, list] = {i: [] for i in range(len(groups))}
    for obs in observations:
        x1, y1, x2, y2 = obs.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for i, g in enumerate(groups):
            gx1, gy1, gx2, gy2 = g.bbox
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                buckets[i].append(obs)
                break

    out: list[BubbleGroup] = []
    for i, g in enumerate(groups):
        matched = buckets[i]
        if matched:
            matched.sort(key=lambda o: ((o.bbox[1] + o.bbox[3]) // 2, o.bbox[0]))
            text = " ".join(o.text for o in matched).strip()
            confidence = min(o.confidence for o in matched)
        else:
            text = ""
            confidence = 0.0
        out.append(_replace(g, text=text, confidence=confidence))
    return tuple(out)


def _replace(g: BubbleGroup, **kwargs) -> BubbleGroup:
    from dataclasses import replace
    return replace(g, **kwargs)
