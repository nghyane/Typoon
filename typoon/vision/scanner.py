"""Page scanner — PP-OCR det + platform OCR.

Detection: PP-OCR det (shared, all platforms).
Recognition: Apple Vision (macOS) or Tesseract (Win/Server).
Both OCR backends work on whole-bubble crops.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from .types import TextMask, TextRegion

_PPOCR_MAX_TILE_HEIGHT = 2048


@dataclass
class ScannedBubble:
    """A merged text bubble with recognized text."""

    polygon: list[list[float]]
    text: str
    confidence: float
    masks: list[TextMask] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# Base scanner — shared detection, subclass provides OCR
# ═════════════════════════════════════════════════════════════════════


class _BaseScanner:
    """PP-OCR det → merge → OCR (subclass). Shared detection logic."""

    def __init__(self, detector) -> None:
        from .detect import TextDetector
        self._det: TextDetector = detector

    def scan(self, image: np.ndarray) -> list[ScannedBubble]:
        from .merge import group_lines
        from .tiling import compute_tiles, deduplicate_regions, offset_regions

        h, w = image.shape[:2]
        tiles = compute_tiles(h, _PPOCR_MAX_TILE_HEIGHT)

        all_regions: list[TextRegion] = []
        all_probs: list[tuple[int, int, np.ndarray]] = []

        for tile_y, tile_h in tiles:
            tile_img = image[tile_y:tile_y + tile_h]
            output = self._det.detect(tile_img)
            if len(tiles) > 1:
                offset_regions(output.regions, tile_y, image)
            all_regions.extend(output.regions)
            if output.prob_image is not None:
                all_probs.append((tile_y, tile_h, output.prob_image))

        if len(tiles) > 1:
            all_regions = deduplicate_regions(all_regions)

        prob_image = _composite_prob(all_probs, h, w)
        merged = group_lines(all_regions, prob_image, image=image)

        # Crop whole bubbles for OCR
        bubble_crops = []
        bubble_indices = []
        for i, mb in enumerate(merged):
            crop = _crop_bubble(image, mb.polygon)
            if crop is not None:
                bubble_crops.append(crop)
                bubble_indices.append(i)

        ocr_results = self._ocr_crops(bubble_crops)

        result: list[ScannedBubble] = []
        for idx, (text, conf) in zip(bubble_indices, ocr_results):
            if not text.strip():
                continue
            mb = merged[idx]
            result.append(ScannedBubble(
                polygon=mb.polygon, text=text.strip(),
                confidence=conf, masks=mb.masks,
            ))
        return result

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
        import cv2
        import Vision
        import Quartz
        import objc
        from CoreFoundation import CFDataCreate

        results: list[tuple[str, float]] = []
        with objc.autorelease_pool():
            for crop in crops:
                ch, cw = crop.shape[:2]
                if ch < 5 or cw < 5:
                    results.append(("", 0.0))
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                data = CFDataCreate(None, gray.tobytes(), len(gray.tobytes()))
                provider = Quartz.CGDataProviderCreateWithCFData(data)
                cg_image = Quartz.CGImageCreate(
                    cw, ch, 8, 8, cw,
                    Quartz.CGColorSpaceCreateDeviceGray(),
                    0,
                    provider, None, False,
                    Quartz.kCGRenderingIntentDefault,
                )

                req = Vision.VNRecognizeTextRequest.alloc().init()
                req.setRecognitionLanguages_(self._languages)
                req.setRecognitionLevel_(0)
                req.setUsesLanguageCorrection_(True)
                handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                    cg_image, None,
                )
                ok, err = handler.performRequests_error_([req], None)
                if err or not req.results():
                    results.append(("", 0.0))
                    continue

                texts = []
                min_conf = 1.0
                for obs in req.results():
                    cand = obs.topCandidates_(1)[0]
                    texts.append(cand.string())
                    min_conf = min(min_conf, float(cand.confidence()))
                results.append((" ".join(texts), min_conf))

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


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _crop_bubble(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray | None:
    """Crop bubble region from image with small padding."""
    h, w = image.shape[:2]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    pad = int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1)
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return image[y1:y2, x1:x2]


def _composite_prob(
    probs: list[tuple[int, int, np.ndarray]], h: int, w: int,
) -> np.ndarray | None:
    if not probs:
        return None
    prob_image = np.zeros((h, w), dtype=np.uint8)
    for tile_y, _, prob in probs:
        ph, pw = prob.shape[:2]
        prob_image[tile_y:tile_y + ph, :pw] = np.maximum(
            prob_image[tile_y:tile_y + ph, :pw], prob[:, :pw],
        )
    return prob_image


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
