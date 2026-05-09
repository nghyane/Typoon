"""OCR backends — page-level and crop-level recognizers.

Two protocols:
- `PageOcr`: recognize a full page image, return positioned text observations.
- `CropOcr`: recognize a list of cropped regions, return text per crop.

Apple Vision, Windows OCR, Tesseract, and Google Lens all expose page-level
text+geometry, so they implement `PageOcr`. The vision pipeline maps each
observation to a detected group by `raw_bbox` containment.

manga-ocr (Japanese) takes a single image and returns one string with no
geometry — it implements `CropOcr` and is fed pre-cropped bubbles.

The factory routes by source language and config: lang=ja → manga-ocr;
otherwise the configured page backend (default Apple Vision on macOS,
Tesseract elsewhere).
"""

from .types import CropOcr, Observation, PageOcr
from .factory import create_ocr

__all__ = ["CropOcr", "Observation", "PageOcr", "create_ocr"]
