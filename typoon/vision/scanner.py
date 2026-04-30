"""Page scanner — PP-OCR text units + optional scope grouping + OCR.

Detection: PP-OCR det (shared, all platforms).
Recognition: Apple Vision (macOS) or Tesseract (Win/Server).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .types import VisualTextGroup


# ═════════════════════════════════════════════════════════════════════
# Base scanner — shared detection, subclass provides OCR
# ═════════════════════════════════════════════════════════════════════


class _BaseScanner:
    """PP-OCR det plus platform OCR. Grouping lives in text_grouping."""

    def __init__(self, detector) -> None:
        from .detect import TextDetector
        self._det: TextDetector = detector

    def scan(
        self,
        image: np.ndarray,
        *,
        scope_model: Any | None = None,
        scope_imgsz: int = 640,
        scope_conf: float = 0.3,
    ) -> list[VisualTextGroup]:
        """PP-OCR detect → YOLO scope → heuristic subgroup → OCR group crops."""
        from .text_grouping import group_and_ocr

        return group_and_ocr(
            self,
            image,
            yolo_model=scope_model,
            yolo_imgsz=scope_imgsz,
            yolo_conf=scope_conf,
        )

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Subclass implements: OCR a list of bubble crops."""
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════
# Apple Vision OCR (macOS)
# ═════════════════════════════════════════════════════════════════════


def _vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        return True
    except ImportError:
        return False


class VisionScanner(_BaseScanner):
    """PP-OCR det + Apple Vision OCR."""

    def __init__(self, detector, languages: list[str] | None = None) -> None:
        super().__init__(detector)
        self._languages = languages or ["en-US"]

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import Vision
        import Quartz
        import objc
        import ctypes
        from AppKit import NSBitmapImageRep
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _ocr_one(crop: np.ndarray) -> tuple[str, float]:
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                return ("", 0.0)
            with objc.autorelease_pool():
                # Fast path: RGB → NSBitmapImageRep (single memcpy, no grayscale conversion)
                if not crop.flags['C_CONTIGUOUS']:
                    crop = np.ascontiguousarray(crop)

                rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel_(
                    (None, None, None, None, None),
                    cw, ch,
                    8,           # bitsPerSample
                    3,           # samplesPerPixel (RGB)
                    False,       # hasAlpha
                    False,       # isPlanar
                    'NSDeviceRGBColorSpace',
                    0,           # bitmapFormat
                    cw * 3,      # bytesPerRow
                    24,          # bitsPerPixel
                )

                # Fast memcpy from numpy buffer into NSBitmapImageRep (single copy)
                dst_arr = np.frombuffer(rep.bitmapData(), dtype=np.uint8)
                dst_ptr = dst_arr.ctypes.data_as(ctypes.c_void_p)
                src_ptr = crop.ctypes.data_as(ctypes.c_void_p)
                ctypes.memmove(dst_ptr, src_ptr, ch * cw * 3)

                ciimage = Quartz.CIImage.alloc().initWithBitmapImageRep_(rep)
                req = Vision.VNRecognizeTextRequest.alloc().init()
                req.setRecognitionLanguages_(self._languages)
                req.setRecognitionLevel_(0)
                req.setUsesLanguageCorrection_(True)
                handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                    ciimage, None,
                )
                ok, err = handler.performRequests_error_([req], None)
                if err or not req.results():
                    return ("", 0.0)
                texts = []
                min_conf = 1.0
                for obs in req.results():
                    cand = obs.topCandidates_(1)[0]
                    texts.append(cand.string())
                    min_conf = min(min_conf, float(cand.confidence()))
                return (" ".join(texts), min_conf)

        if len(crops) <= 1:
            return [_ocr_one(c) for c in crops]

        # Parallel OCR — Apple Vision is thread-safe
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        with ThreadPoolExecutor(max_workers=min(4, len(crops))) as pool:
            futures = {pool.submit(_ocr_one, crop): i for i, crop in enumerate(crops)}
            for f in as_completed(futures):
                try:
                    results[futures[f]] = f.result()
                except Exception:
                    pass  # fallback ("", 0.0) already in place
        return results


# ═════════════════════════════════════════════════════════════════════
# Windows OCR (Windows 10+)
# ═════════════════════════════════════════════════════════════════════


def _winocr_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from winrt.windows.media.ocr import OcrEngine as _  # noqa: F401
        from winrt.windows.graphics.imaging import SoftwareBitmap as _  # noqa: F401
        return True
    except ImportError:
        return False


class WindowsOcrScanner(_BaseScanner):
    """PP-OCR det + Windows.Media.Ocr."""

    def __init__(self, detector, lang: str = "en") -> None:
        super().__init__(detector)
        self._lang = lang

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import asyncio
        import cv2
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import (
            SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode,
        )
        from winrt.windows.security.cryptography import CryptographicBuffer
        from winrt.windows.globalization import Language

        engine = OcrEngine.try_create_from_language(Language(self._lang))
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()

        results: list[tuple[str, float]] = []
        for crop in crops:
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                results.append(("", 0.0))
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            buf = CryptographicBuffer.create_from_byte_array(gray.tobytes())
            bitmap = SoftwareBitmap.create_copy_from_buffer(
                buf, BitmapPixelFormat.GRAY8, cw, ch, BitmapAlphaMode.IGNORE,
            )

            ocr_result = asyncio.run(engine.recognize_async(bitmap))
            text = ocr_result.text.strip() if ocr_result else ""
            text = " ".join(text.split())
            results.append((text, 0.95 if text else 0.0))

        return results


# ═════════════════════════════════════════════════════════════════════
# Tesseract OCR (Linux / fallback)
# ═════════════════════════════════════════════════════════════════════


class TesseractScanner(_BaseScanner):
    """PP-OCR det + Tesseract OCR."""

    def __init__(self, detector, lang: str = "eng") -> None:
        super().__init__(detector)
        self._lang = lang

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import pytesseract

        results: list[tuple[str, float]] = []
        for crop in crops:
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                results.append(("", 0.0))
                continue
            text = pytesseract.image_to_string(
                crop, lang=self._lang, config="--psm 6",
            ).strip()
            text = " ".join(text.split())
            conf = 0.9 if text else 0.0
            results.append((text, conf))
        return results


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def create_scanner(hub=None, languages: list[str] | None = None):
    """Create the best available scanner for this platform.

    Priority: Apple Vision (macOS) > Windows OCR (Win10+) > Tesseract.
    PP-OCR det is always used for detection.
    """
    if hub is None:
        raise RuntimeError("PP-OCR models required")
    from .detect import TextDetector

    detector = TextDetector(hub.resolve("ppocr-det.safetensors"), hub.resolve("ppocr-det-config.json"))

    if _vision_available():
        return VisionScanner(detector, languages=languages)

    if _winocr_available():
        return WindowsOcrScanner(detector)

    if _tesseract_available():
        return TesseractScanner(detector)

    raise RuntimeError(
        "No OCR backend available. Install Tesseract: "
        "apt install tesseract-ocr (Linux) or choco install tesseract (Windows)"
    )
