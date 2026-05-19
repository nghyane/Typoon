"""Windows OCR — page-level via `winrt.windows.media.ocr.OcrEngine`.

The Windows Runtime OCR engine accepts a `SoftwareBitmap` of any size
(internally resamples) and returns `OcrResult.lines`, each line carrying
its own `BoundingRect`. Quality sits between Tesseract and Apple Vision
on stylised fonts; useful as the native Windows fallback when Google
Lens (online) is not desired.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from .types import Observation


logger = logging.getLogger(__name__)

_LANG_MAP: dict[str, str] = {
    "en":    "en",
    "ja":    "ja",
    "ko":    "ko",
    "zh":    "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
    "vi":    "vi",
}


def is_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from winrt.windows.media.ocr import OcrEngine  # noqa: F401
        return True
    except ImportError:
        return False


class WindowsOcrPageOcr:
    def ocr_page(
        self,
        image: np.ndarray,
        *,
        lang: str | None = None,
    ) -> list[Observation]:
        if image.size == 0:
            return []
        import asyncio
        import cv2
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import (
            SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode,
        )
        from winrt.windows.security.cryptography import CryptographicBuffer
        from winrt.windows.globalization import Language

        win_lang = _LANG_MAP.get((lang or "en").lower(), "en")
        engine = OcrEngine.try_create_from_language(Language(win_lang))
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            logger.warning("windows OCR: no engine available for lang=%s", win_lang)
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        buf = CryptographicBuffer.create_from_byte_array(gray.tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            buf, BitmapPixelFormat.GRAY8, w, h, BitmapAlphaMode.IGNORE,
        )
        try:
            result = asyncio.run(engine.recognize_async(bitmap))
        except Exception as e:
            logger.warning("windows OCR failed: %s", e)
            return []
        if result is None:
            return []

        out: list[Observation] = []
        for line in result.lines:
            text = (line.text or "").strip()
            if not text:
                continue
            rect = line.bounding_rect
            x1 = int(rect.x)
            y1 = int(rect.y)
            x2 = int(rect.x + rect.width)
            y2 = int(rect.y + rect.height)
            out.append(Observation(bbox=(x1, y1, x2, y2), text=text, confidence=1.0))
        return out
