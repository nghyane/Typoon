"""PP-OCR DBNet detector — text region detector for offline path.

Wraps the existing CoreML/MLX/ONNX backend (in vision/runtime/ppocr_det/)
in an async interface. Backend releases the GIL during inference so
asyncio.to_thread provides real concurrency.

Detection-only: emits TextBlock with text=None and a polygon. Use a
TextRecognizer downstream to fill in text.
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pyclipper

from ..contracts import DetectionResult, TextBlock


__all__ = ["PPOCRDetector"]

logger = logging.getLogger(__name__)


# ─── Tunables (carried over from legacy detect.py) ────────────────────────


_DET_RESIZE_LONG = 1280
_DET_RESIZE_LONG_SMALL = 1280
_DET_SMALL_THRESH = 960
_DET_MIN_WIDTH = 768
_DET_MAX_PIXELS = 1_500_000
_DET_MAX_DIM = 2048
_DET_THRESH = 0.3
_DET_BOX_THRESH = 0.6
_DET_UNCLIP_RATIO = 1.5
_DET_MIN_SIZE = 5.0
_DET_MAX_CANDIDATES = 1000

# Tile config (was vision/tiling.py)
_TILE_SIZE = 1280
_TILE_OVERLAP = 400


# ─── Detector ─────────────────────────────────────────────────────────────


class PPOCRDetector:
    """PP-OCR DBNet text region detector.

    Loads the safetensors weights via the platform-appropriate backend
    (CoreML on macOS, ONNX elsewhere). Construction is cheap; first
    detect() pays model load cost.
    """

    name = "ppocr_dbnet"

    def __init__(self, model_path: Path | str, config_path: Path | str) -> None:
        from .._backends.ppocr_det import TextDetector as _Backend
        self._backend = _Backend(str(model_path), str(config_path))

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        # Backend runs CoreML/ONNX → GIL released. to_thread provides true
        # parallelism when scan_chapter dispatches multiple pages.
        blocks = await asyncio.to_thread(self._detect_sync, image)
        h, w = image.shape[:2]
        return DetectionResult(
            blocks=tuple(blocks),
            text_already_recognized=False,
            page_size=(w, h),
        )

    def _detect_sync(self, image: np.ndarray) -> list[TextBlock]:
        h, w = image.shape[:2]
        all_blocks: list[TextBlock] = []
        for tx, ty, tw, th in _compute_tiles(h, w):
            tile = image[ty:ty + th, tx:tx + tw]
            tile_blocks = self._detect_tile(tile, tx, ty)
            all_blocks.extend(tile_blocks)
        return _dedup(all_blocks)

    def _detect_tile(
        self, tile: np.ndarray, offset_x: int, offset_y: int,
    ) -> list[TextBlock]:
        padded, cw, ch, pw, ph = _ppocr_preprocess(tile)
        prob = self._backend.detect(padded)  # (ph, pw) float32
        polys = _db_postprocess(prob, pw, ph, cw, ch, tile.shape[1], tile.shape[0])
        out: list[TextBlock] = []
        for polygon, conf in polys:
            shifted = tuple((p[0] + offset_x, p[1] + offset_y) for p in polygon)
            xs = [p[0] for p in shifted]
            ys = [p[1] for p in shifted]
            bbox = (
                int(min(xs)), int(min(ys)),
                int(max(xs)), int(max(ys)),
            )
            out.append(TextBlock(
                bbox=bbox,
                polygon=shifted,
                confidence=conf,
                text=None,
                detector=self.name,
            ))
        return out


# ─── Tiling ───────────────────────────────────────────────────────────────


def _compute_tiles(h: int, w: int) -> list[tuple[int, int, int, int]]:
    stride = _TILE_SIZE - _TILE_OVERLAP
    ys: list[int] = []
    y = 0
    while y < h:
        ys.append(y)
        if y + _TILE_SIZE >= h:
            break
        y += stride
    xs: list[int] = []
    x = 0
    while x < w:
        xs.append(x)
        if x + _TILE_SIZE >= w:
            break
        x += stride
    out: list[tuple[int, int, int, int]] = []
    for y in ys:
        th = min(_TILE_SIZE, h - y)
        for x in xs:
            tw = min(_TILE_SIZE, w - x)
            out.append((x, y, tw, th))
    return out


def _dedup(blocks: list[TextBlock]) -> list[TextBlock]:
    """IoU-based dedup across overlapping tiles. Keep higher confidence."""
    if len(blocks) <= 1:
        return blocks

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(1, aa + ab - inter)

    keep = [True] * len(blocks)
    for i in range(len(blocks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(blocks)):
            if not keep[j]:
                continue
            if _iou(blocks[i].bbox, blocks[j].bbox) > 0.5:
                if blocks[i].confidence >= blocks[j].confidence:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [b for b, k in zip(blocks, keep) if k]


# ─── DBNet preprocessing + postprocessing (carry-over from legacy) ────────


def _ppocr_preprocess(image: np.ndarray):
    h, w = image.shape[:2]
    long = _DET_RESIZE_LONG_SMALL if max(w, h) < _DET_SMALL_THRESH else _DET_RESIZE_LONG
    ratio = max(long / max(w, h), _DET_MIN_WIDTH / w)
    nw, nh = int(w * ratio), int(h * ratio)

    if nw * nh > _DET_MAX_PIXELS:
        s = math.sqrt(_DET_MAX_PIXELS / (nw * nh))
        nw, nh = int(nw * s), int(nh * s)

    dim_cap = (_DET_MAX_DIM // 32) * 32
    nw = min(nw, dim_cap)
    nh = min(nh, dim_cap)

    resized = cv2.resize(image, (nw, nh))
    pw = ((nw + 31) // 32) * 32
    ph = ((nh + 31) // 32) * 32
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, nw, nh, pw, ph


def _db_postprocess(
    prob: np.ndarray,
    pw: int, ph: int, cw: int, ch: int,
    orig_w: int, orig_h: int,
) -> list[tuple[tuple[tuple[float, float], ...], float]]:
    bitmap = (prob[:ch, :cw] > _DET_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:_DET_MAX_CANDIDATES]

    sx = orig_w / cw
    sy = orig_h / ch
    out: list[tuple[tuple[tuple[float, float], ...], float]] = []
    for c in contours:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        if min(rect[1]) < _DET_MIN_SIZE:
            continue
        box = cv2.boxPoints(rect).astype(np.float32)
        score = _box_score(prob, box)
        if score < _DET_BOX_THRESH:
            continue
        unclipped = _unclip(box)
        if unclipped is None:
            continue
        scaled = tuple(
            (float(p[0]) * sx, float(p[1]) * sy) for p in unclipped
        )
        out.append((scaled, float(score)))
    return out


def _box_score(prob: np.ndarray, box: np.ndarray) -> float:
    h, w = prob.shape
    xmin = max(0, int(np.floor(box[:, 0].min())))
    xmax = min(w - 1, int(np.ceil(box[:, 0].max())))
    ymin = max(0, int(np.floor(box[:, 1].min())))
    ymax = min(h - 1, int(np.ceil(box[:, 1].max())))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
    return float(prob[ymin:ymax + 1, xmin:xmax + 1][mask == 1].mean())


def _unclip(box: np.ndarray) -> list[list[float]] | None:
    poly = box.tolist()
    area = cv2.contourArea(box)
    perim = cv2.arcLength(box, True)
    if perim == 0:
        return None
    distance = area * _DET_UNCLIP_RATIO / perim
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None
    return expanded[0]
