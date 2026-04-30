"""OcrBackend protocol — platform OCR implementations.

Each backend recognizes a batch of RGB crop images and returns
(text, confidence) pairs. Failures return ("", 0.0) rather than raising.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OcrBackend(Protocol):
    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


# ── Apple Vision (macOS) ─────────────────────────────────────────


def _vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision   # noqa: F401
        import Quartz   # noqa: F401
        return True
    except ImportError:
        return False


class AppleVisionBackend:
    """PP-OCR crops → Apple Vision OCR. Thread-safe; batches run in parallel."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self._languages = languages or ["en-US"]

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        if len(crops) == 1:
            return [self._ocr_one(crops[0])]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        with ThreadPoolExecutor(max_workers=min(4, len(crops))) as pool:
            futures = {pool.submit(self._ocr_one, crop): i for i, crop in enumerate(crops)}
            for f in as_completed(futures):
                try:
                    results[futures[f]] = f.result()
                except Exception:
                    pass
        return results

    def _ocr_one(self, crop: np.ndarray) -> tuple[str, float]:
        import Vision
        import Quartz
        import objc
        import ctypes
        from AppKit import NSBitmapImageRep

        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            return ("", 0.0)
        with objc.autorelease_pool():
            if not crop.flags["C_CONTIGUOUS"]:
                crop = np.ascontiguousarray(crop)
            rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel_(
                (None, None, None, None, None),
                cw, ch, 8, 3, False, False,
                "NSDeviceRGBColorSpace", 0, cw * 3, 24,
            )
            dst = np.frombuffer(rep.bitmapData(), dtype=np.uint8)
            ctypes.memmove(
                dst.ctypes.data_as(ctypes.c_void_p),
                crop.ctypes.data_as(ctypes.c_void_p),
                ch * cw * 3,
            )
            ciimage = Quartz.CIImage.alloc().initWithBitmapImageRep_(rep)
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLanguages_(self._languages)
            req.setRecognitionLevel_(0)
            req.setUsesLanguageCorrection_(True)
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ciimage, None)
            ok, err = handler.performRequests_error_([req], None)
            if err or not req.results():
                return ("", 0.0)
            texts, min_conf = [], 1.0
            for obs in req.results():
                cand = obs.topCandidates_(1)[0]
                texts.append(cand.string())
                min_conf = min(min_conf, float(cand.confidence()))
            return (" ".join(texts), min_conf)


# ── Windows OCR ──────────────────────────────────────────────────


def _winocr_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from winrt.windows.media.ocr import OcrEngine as _  # noqa: F401
        return True
    except ImportError:
        return False


class WindowsOcrBackend:
    def __init__(self, lang: str = "en") -> None:
        self._lang = lang

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import asyncio
        import cv2
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
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
            text = " ".join((ocr_result.text.strip() if ocr_result else "").split())
            results.append((text, 0.95 if text else 0.0))
        return results


# ── Tesseract ────────────────────────────────────────────────────


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class TesseractBackend:
    def __init__(self, lang: str = "eng") -> None:
        self._lang = lang

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import pytesseract
        results: list[tuple[str, float]] = []
        for crop in crops:
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                results.append(("", 0.0))
                continue
            text = " ".join(pytesseract.image_to_string(
                crop, lang=self._lang, config="--psm 6",
            ).strip().split())
            results.append((text, 0.9 if text else 0.0))
        return results


# ── Factory ──────────────────────────────────────────────────────


def create_ocr_backend(languages: list[str] | None = None) -> OcrBackend:
    """Return the best available OCR backend for this platform."""
    if _vision_available():
        return AppleVisionBackend(languages)
    if _winocr_available():
        return WindowsOcrBackend()
    if _tesseract_available():
        return TesseractBackend()
    raise RuntimeError(
        "No OCR backend available. "
        "Install Tesseract: apt install tesseract-ocr (Linux) "
        "or choco install tesseract (Windows)"
    )
