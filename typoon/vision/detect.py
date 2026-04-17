"""Text detection — PP-OCR det (DBNet++) for all languages.

Backends: MLX (Mac) or PyTorch (Windows/Server), safetensors only.

Ported from crates/engine/src/vision/ocr/ppocr.rs (detection + postprocess).
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .types import DetectionOutput, TextMask, TextRegion

logger = logging.getLogger(__name__)

try:
    import pyclipper as _pyclipper
except ImportError:
    _pyclipper = None  # type: ignore[assignment]

# ── PP-OCR detection constants ───────────────────────────────────────

_DET_RESIZE_LONG = 960
_DET_RESIZE_LONG_SMALL = 1280
_DET_SMALL_THRESH = 960
_DET_MIN_WIDTH = 768
_DET_MAX_PIXELS = 1_500_000
_DET_THRESH = 0.3
_DET_BOX_THRESH = 0.6
_DET_UNCLIP_RATIO = 1.5
_DET_MIN_SIZE = 5.0
_DET_MAX_CANDIDATES = 1000


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════


class TextDetector:
    """PP-OCR text detector for all languages. Uses safetensors + MLX/PyTorch."""

    def __init__(self, model_path: str, config_path: str) -> None:
        from .runtime.ppocr_det import TextDetector as _Backend
        self._backend = _Backend(model_path, config_path)

    def detect(self, image: np.ndarray) -> DetectionOutput:
        """Detect text regions. image: RGB uint8 HWC."""
        orig_h, orig_w = image.shape[:2]
        padded, cw, ch, pw, ph = _ppocr_preprocess(image)

        prob = self._backend.detect(padded)  # [ph, pw] float32

        prob_image = _build_prob_image(prob, cw, ch, orig_w, orig_h)
        regions = _db_postprocess(prob, pw, ph, cw, ch, orig_w, orig_h, image)

        logger.debug("PP-OCR detected %d text regions", len(regions))
        return DetectionOutput(regions=regions, prob_image=prob_image)


# ═════════════════════════════════════════════════════════════════════
# PP-OCR DBNet internals
# ═════════════════════════════════════════════════════════════════════


def _ppocr_preprocess(image: np.ndarray):
    """Resize + pad to multiple of 32. Returns (padded_rgb_uint8, cw, ch, pw, ph)."""
    h, w = image.shape[:2]

    long = _DET_RESIZE_LONG_SMALL if max(w, h) < _DET_SMALL_THRESH else _DET_RESIZE_LONG
    ratio = max(long / max(w, h), _DET_MIN_WIDTH / w)
    nw, nh = int(w * ratio), int(h * ratio)

    if nw * nh > _DET_MAX_PIXELS:
        s = math.sqrt(_DET_MAX_PIXELS / (nw * nh))
        nw, nh = int(nw * s), int(nh * s)

    pw = ((nw + 31) // 32) * 32
    ph = ((nh + 31) // 32) * 32

    resized = cv2.resize(image, (nw, nh))
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:nh, :nw] = resized

    return padded, nw, nh, pw, ph


def _build_prob_image(
    prob: np.ndarray, content_w: int, content_h: int,
    orig_w: int, orig_h: int,
) -> np.ndarray:
    content = prob[:content_h, :content_w]
    u8 = np.clip(content * 255, 0, 255).astype(np.uint8)
    return cv2.resize(u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _db_postprocess(
    prob: np.ndarray,
    pad_w: int, pad_h: int,
    content_w: int, content_h: int,
    orig_w: int, orig_h: int,
    image: np.ndarray,
) -> list[TextRegion]:
    binary = (prob > _DET_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    sx, sy = orig_w / content_w, orig_h / content_h
    regions: list[TextRegion] = []

    for contour in contours[:_DET_MAX_CANDIDATES]:
        if len(contour) < 3:
            continue

        pts = contour.reshape(-1, 2)
        score = float(np.mean([
            prob[min(y, pad_h - 1), min(x, pad_w - 1)] for x, y in pts
        ]))
        if score < _DET_BOX_THRESH:
            continue

        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if min(w, h) < _DET_MIN_SIZE:
            continue

        # Unclip
        area, peri = w * h, 2 * (w + h)
        if peri < 1e-6:
            continue
        expanded = _unclip(contour, area * _DET_UNCLIP_RATIO / peri)
        if expanded is None:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(expanded))
        corners = _order_corners(box)

        # Scale to original
        orig_corners = np.array([
            [np.clip(x * sx, 0, orig_w - 1), np.clip(y * sy, 0, orig_h - 1)]
            for x, y in corners
        ], dtype=np.float32)

        crop = _warp_crop(image, orig_corners)
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        polygon = orig_corners.tolist()
        mask = _build_ppocr_mask(prob, content_w, content_h, orig_w, orig_h, orig_corners, polygon)

        regions.append(TextRegion(
            polygon=polygon, crop=crop, confidence=score, mask=mask,
        ))

    return regions


def _unclip(contour: np.ndarray, d: float) -> np.ndarray | None:
    """Expand contour by distance d."""
    if _pyclipper is not None:
        poly = contour.reshape(-1, 2).tolist()
        pco = _pyclipper.PyclipperOffset()
        pco.AddPath(poly, _pyclipper.JT_ROUND, _pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco.Execute(d)
        if not expanded:
            return None
        return np.array(expanded[0], dtype=np.int32).reshape(-1, 1, 2)

    logger.debug("pyclipper not available, using centroid fallback")
    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    result = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            result.append([x, y])
        else:
            result.append([x + d * dx / dist, y + d * dy / dist])
    return np.array(result, dtype=np.int32).reshape(-1, 1, 2)


def _order_corners(box: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    return np.array([
        box[np.argmin(s)],   # TL
        box[np.argmin(diff)], # TR
        box[np.argmax(s)],   # BR
        box[np.argmax(diff)], # BL
    ], dtype=np.float32)


def _warp_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    tl, tr, br, bl = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if out_w < 2 or out_h < 2:
        return None

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    crop = cv2.warpPerspective(image, M, (out_w, out_h))

    if out_h / max(out_w, 1) >= 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def _build_ppocr_mask(
    prob: np.ndarray,
    content_w: int, content_h: int,
    orig_w: int, orig_h: int,
    corners: np.ndarray,
    polygon: list[list[float]],
) -> TextMask | None:
    xs, ys = corners[:, 0], corners[:, 1]
    mx = max(0, int(math.floor(xs.min())))
    my = max(0, int(math.floor(ys.min())))
    mw = max(1, min(int(math.ceil(xs.max())) - mx, orig_w - mx))
    mh = max(1, min(int(math.ceil(ys.max())) - my, orig_h - my))

    # Sample prob at detection resolution, bilinear resize
    inv_sx, inv_sy = content_w / orig_w, content_h / orig_h
    dx1 = int(math.floor(mx * inv_sx))
    dy1 = int(math.floor(my * inv_sy))
    dx2 = min(int(math.ceil((mx + mw) * inv_sx)), prob.shape[1])
    dy2 = min(int(math.ceil((my + mh) * inv_sy)), prob.shape[0])

    dm_crop = prob[dy1:dy2, dx1:dx2]
    dm_u8 = np.clip(dm_crop * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(dm_u8, (mw, mh), interpolation=cv2.INTER_LINEAR)

    mask = (resized > 51).astype(np.uint8) * 255  # threshold ~0.20

    dilate_v = min(mh // 4, 30)
    dilate_h = min(mw // 4, 30)
    if max(dilate_v, dilate_h) > 0 and np.any(mask):
        kh = dilate_v * 2 + 1
        kw = dilate_h * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))

    # Clip to polygon using cv2.fillPoly
    _clip_mask_to_polygon(mask, mx, my, polygon)

    return TextMask(x=mx, y=my, image=mask)


def _clip_mask_to_polygon(
    mask: np.ndarray, mask_x: int, mask_y: int, polygon: list[list[float]],
) -> None:
    """Zero out mask pixels outside the polygon."""
    if len(polygon) < 3:
        return
    local = np.round(
        np.array(polygon, dtype=np.float32) - [mask_x, mask_y]
    ).astype(np.int32)
    clip = np.zeros_like(mask)
    cv2.fillPoly(clip, [local], 255)
    mask &= clip
