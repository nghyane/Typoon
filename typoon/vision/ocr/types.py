"""Public OCR types — protocols + observation record."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class Observation(NamedTuple):
    """One recognized text region in absolute page coordinates.

    `bbox` is (x1, y1, x2, y2) in page pixels. `text` is the recognized
    string for the region. `confidence` is the backend's per-region score
    in [0, 1]; backends that don't expose a score return 1.0.
    """
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


@runtime_checkable
class PageOcr(Protocol):
    """Recognize text on a full page, return positioned observations.

    The pipeline assigns each observation to a detected group by checking
    whether the observation's center falls inside the group's `raw_bbox`.
    Implementations may tile the image internally; observation coordinates
    must be in absolute page pixels.
    """
    def ocr_page(
        self,
        image: np.ndarray,
        *,
        lang: str | None = None,
    ) -> list[Observation]: ...


@runtime_checkable
class CropOcr(Protocol):
    """Recognize text on a batch of pre-cropped regions.

    Used for backends that accept one image and return one string with no
    geometry (e.g. manga-ocr). The pipeline feeds raw_bbox crops in order;
    the returned (text, confidence) list aligns 1:1 with the input crops.
    """
    def ocr_crops(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]: ...
