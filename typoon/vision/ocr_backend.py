"""OcrBackend protocol — platform OCR implementations.

Each backend recognizes a batch of RGB crop images and returns
(text, confidence) pairs. Failures return ("", 0.0) rather than raising.

Language is passed per call (not per construction): one VisionRuntime
serves projects in different `source_lang`s, so binding the recognizer
to a single language at startup would silently force every project
through the wrong recognizer.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OcrBackend(Protocol):
    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]: ...


# Backends set `wants_raw = True` (class attribute) to opt out of the
# adaptive-threshold preprocessing the pipeline applies to group crops.
# manga-ocr was trained on natural grayscale manga and is degraded by
# binarization; Apple/Win/Tesseract benefit from it.


# ── Language code mapping ────────────────────────────────────────
#
# Project source_lang uses ISO 639-1 ("ja", "ko", "zh", "en"). Each
# backend wants its own format. One project = one source language;
# off-script SFX (e.g. Korean SFX in an English chapter) get low conf
# and fall out of the noise filter naturally — that's expected.

_APPLE_VISION_LANGS: dict[str, list[str]] = {
    "ja":    ["ja-JP"],
    "ko":    ["ko-KR"],
    "zh":    ["zh-Hans"],
    "zh-cn": ["zh-Hans"],
    "zh-tw": ["zh-Hant"],
    "en":    ["en-US"],
    "vi":    ["vi-VN"],
}

_WINDOWS_OCR_LANGS: dict[str, str] = {
    "ja":    "ja",
    "ko":    "ko",
    "zh":    "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
    "en":    "en",
    "vi":    "vi",
}

_TESSERACT_LANGS: dict[str, str] = {
    "ja":    "jpn",
    "ko":    "kor",
    "zh":    "chi_sim",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "en":    "eng",
    "vi":    "vie",
}


def _normalize(lang: str | None) -> str:
    return (lang or "en").lower()


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

    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]:
        if not crops:
            return []
        languages = _APPLE_VISION_LANGS.get(_normalize(lang), ["en-US"])
        if len(crops) == 1:
            return [self._ocr_one(crops[0], languages)]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        with ThreadPoolExecutor(max_workers=min(4, len(crops))) as pool:
            futures = {
                pool.submit(self._ocr_one, crop, languages): i
                for i, crop in enumerate(crops)
            }
            for f in as_completed(futures):
                try:
                    results[futures[f]] = f.result()
                except Exception:
                    pass
        return results

    def _ocr_one(self, crop: np.ndarray, languages: list[str]) -> tuple[str, float]:
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
            req.setRecognitionLanguages_(languages)
            req.setRecognitionLevel_(0)  # accurate
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
    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]:
        import asyncio
        import cv2
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
        from winrt.windows.security.cryptography import CryptographicBuffer
        from winrt.windows.globalization import Language

        win_lang = _WINDOWS_OCR_LANGS.get(_normalize(lang), "en")
        engine = OcrEngine.try_create_from_language(Language(win_lang))
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
    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]:
        import pytesseract
        tess_lang = _TESSERACT_LANGS.get(_normalize(lang), "eng")
        results: list[tuple[str, float]] = []
        for crop in crops:
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                results.append(("", 0.0))
                continue
            text = " ".join(pytesseract.image_to_string(
                crop, lang=tess_lang, config="--psm 6",
            ).strip().split())
            results.append((text, 0.9 if text else 0.0))
        return results


# ── Factory ──────────────────────────────────────────────────────


class _RoutingBackend:
    """Routes per-call to a language-specific backend, falling back otherwise.

    manga-ocr-base wins decisively on Japanese manga (vertical text, furigana,
    SFX). All other languages stay on the platform default backend, which is
    fine for Latin / CJK printed text that platform OCR handles well.
    """

    def __init__(self, fallback: OcrBackend, ja: OcrBackend | None) -> None:
        self._fallback = fallback
        self._ja = ja

    def _route(self, lang: str | None) -> OcrBackend:
        if self._ja is not None and _normalize(lang) == "ja":
            return self._ja
        return self._fallback

    def recognize(
        self,
        crops: list[np.ndarray],
        *,
        lang: str | None = None,
    ) -> list[tuple[str, float]]:
        return self._route(lang).recognize(crops, lang=lang)

    def wants_raw(self, lang: str | None) -> bool:
        return bool(getattr(self._route(lang), "wants_raw", False))


def create_ocr_backend() -> OcrBackend:
    """Return the best available OCR backend for this platform.

    Japanese is routed to manga-ocr-base when available; other languages
    use the platform default. Language is selected per call, not per
    backend instance.
    """
    if _vision_available():
        fallback: OcrBackend = AppleVisionBackend()
    elif _winocr_available():
        fallback = WindowsOcrBackend()
    elif _tesseract_available():
        fallback = TesseractBackend()
    else:
        raise RuntimeError(
            "No OCR backend available. "
            "Install Tesseract: apt install tesseract-ocr (Linux) "
            "or choco install tesseract (Windows)"
        )

    ja: OcrBackend | None = None
    try:
        from .ocr_manga import MangaOcrBackend, _manga_ocr_available
        if _manga_ocr_available():
            ja = MangaOcrBackend()
    except ImportError:
        pass

    return _RoutingBackend(fallback, ja)
