"""Apple Vision OCR — page-level via vertical tiling.

Apple Vision (`VNRecognizeTextRequest`) handles tall images poorly: it
silently returns zero observations once the image exceeds roughly
5000–6000 pixels in either dimension. Manhwa pages are ~720×10000, so we
slice vertically into overlapping tiles, OCR each, and combine.

Tile parameters are chosen from measured behaviour, not from heuristics:
- `_TILE`: 3000px keeps every tile well below the silent failure cliff.
- `_OVERLAP`: 500px is larger than any plausible single bubble height
  (manhwa bubbles cluster at 100–250px), so a bubble that straddles a
  tile boundary appears intact in at least one tile.
- Sequential dispatch: Apple Vision parallelises internally across CPU
  cores. A `ThreadPoolExecutor` over tiles gave no speed-up in
  benchmarks (270ms sequential vs 270–320ms parallel) and adds
  complexity, so tiles run one after another.

Observations are deduplicated across overlap regions by IoU on the
absolute-page bbox: two observations that overlap by ≥50% of their own
area are treated as the same text, and the higher-confidence copy wins.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .types import Observation


_TILE = 3000
_OVERLAP = 500
_DEDUP_IOU = 0.5

_LANG_MAP: dict[str, list[str]] = {
    "en":    ["en-US"],
    "ja":    ["ja-JP"],
    "ko":    ["ko-KR"],
    "zh":    ["zh-Hans"],
    "zh-cn": ["zh-Hans"],
    "zh-tw": ["zh-Hant"],
    "vi":    ["vi-VN"],
}


def is_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision   # noqa: F401
        import Quartz   # noqa: F401
        return True
    except ImportError:
        return False


class AppleVisionPageOcr:
    """Tile + sequential OCR via `VNRecognizeTextRequest`."""

    def ocr_page(
        self,
        image: np.ndarray,
        *,
        lang: str | None = None,
    ) -> list[Observation]:
        if image.size == 0:
            return []
        languages = _LANG_MAP.get((lang or "en").lower(), ["en-US"])
        height = image.shape[0]
        step = _TILE - _OVERLAP

        raw: list[Observation] = []
        y = 0
        while y < height:
            y_end = min(y + _TILE, height)
            if y_end - y < 100:
                break
            tile = image[y:y_end]
            for obs in _ocr_tile(tile, languages):
                # Lift tile-relative bbox into absolute page coordinates.
                x1, ty1, x2, ty2 = obs.bbox
                raw.append(Observation(
                    bbox=(x1, y + ty1, x2, y + ty2),
                    text=obs.text,
                    confidence=obs.confidence,
                ))
            if y_end == height:
                break
            y += step

        return _dedup(raw)


def _ocr_tile(tile: np.ndarray, languages: list[str]) -> list[Observation]:
    import Vision
    import Quartz
    import objc
    import ctypes
    from AppKit import NSBitmapImageRep

    h, w = tile.shape[:2]
    if h < 5 or w < 5:
        return []
    if not tile.flags["C_CONTIGUOUS"]:
        tile = np.ascontiguousarray(tile)

    with objc.autorelease_pool():
        rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel_(
            (None, None, None, None, None),
            w, h, 8, 3, False, False,
            "NSDeviceRGBColorSpace", 0, w * 3, 24,
        )
        dst = np.frombuffer(rep.bitmapData(), dtype=np.uint8)
        ctypes.memmove(
            dst.ctypes.data_as(ctypes.c_void_p),
            tile.ctypes.data_as(ctypes.c_void_p),
            h * w * 3,
        )
        ciimage = Quartz.CIImage.alloc().initWithBitmapImageRep_(rep)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLanguages_(languages)
        req.setRecognitionLevel_(0)             # accurate, not fast
        req.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ciimage, None)
        ok, err = handler.performRequests_error_([req], None)
        if err or not req.results():
            return []

        out: list[Observation] = []
        for vn_obs in req.results():
            cand = vn_obs.topCandidates_(1)[0]
            text = cand.string()
            if not text:
                continue
            # Vision uses normalized, origin-bottom-left coordinates.
            bb = vn_obs.boundingBox()
            x1 = int(bb.origin.x * w)
            x2 = int((bb.origin.x + bb.size.width) * w)
            y1 = int((1.0 - bb.origin.y - bb.size.height) * h)
            y2 = int((1.0 - bb.origin.y) * h)
            out.append(Observation(
                bbox=(x1, y1, x2, y2),
                text=text,
                confidence=float(cand.confidence()),
            ))
        return out


def _dedup(observations: list[Observation]) -> list[Observation]:
    """Drop observations duplicated across overlap regions by IoU."""
    # Sort by confidence descending so the better copy survives.
    sorted_obs = sorted(observations, key=lambda o: -o.confidence)
    kept: list[Observation] = []
    for obs in sorted_obs:
        if any(_iou_ratio(obs.bbox, k.bbox) > _DEDUP_IOU for k in kept):
            continue
        kept.append(obs)
    return kept


def _iou_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection area divided by smaller box area — symmetric for our use."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(aa, bb)
