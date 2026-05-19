"""Comic Text Detector — ONNX/CoreML inference backend.

Uses the pre-exported ONNX model from mayocream/comic-text-detector-onnx.
CoreML execution provider is used automatically on macOS (Apple Silicon),
falling back to CPU. Single model file, no architecture reimplementation.

Output:
  blk  (1, 64512, 7) — YOLOv5 decoded: [cx, cy, w, h, obj, cls_text, cls_bubble]
                        class 0 = bubble, class 1 = text (reversed vs Koharu)
  seg  (1, 1, H, W)  — UNet bubble segmentation mask
  det  (1, 2, H, W)  — DBNet [shrink_map, threshold_map]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_DETECT_SIZE        = 1024
_CONF_THRESHOLD     = 0.35
_NMS_THRESHOLD      = 0.35
_DBNET_K            = 50.0
_BINARY_THRESHOLD   = 0.30
_DILATION_RADIUS    = 3
_HOLE_CLOSE_RADIUS  = 10
_BBOX_DILATION      = 1.0
# class indices in blk output (reversed vs Koharu Rust impl)
_CLS_TEXT   = 1
_CLS_BUBBLE = 0


@dataclass(slots=True)
class TextRegion:
    bbox:       tuple[int, int, int, int]
    polygon:    tuple[tuple[float, float], ...]
    confidence: float
    text_mask:  np.ndarray   # uint8 (H, W) page-coords crop
    mask_x:     int
    mask_y:     int


@dataclass(slots=True)
class CTDResult:
    text_regions: list[TextRegion]
    bubble_mask:  np.ndarray   # uint8 (H, W) full page


class CTDBackend:
    """ONNX CTD backend — CoreML on macOS, CPU fallback."""

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def detect(self, image: np.ndarray) -> CTDResult:
        """Run full CTD pipeline. image: RGB uint8 (H, W, 3)."""
        orig_h, orig_w = image.shape[:2]
        inp, rw, rh = _preprocess(image)

        blk_out, seg_out, det_out = self._sess.run(None, {"images": inp})

        bboxes      = _decode_yolo(blk_out[0], orig_w, orig_h, rw, rh)
        bubble_mask = _decode_seg(seg_out[0, 0], orig_w, orig_h, rw, rh)
        text_mask   = _decode_det_fused(det_out[0], seg_out[0, 0], orig_w, orig_h, rw, rh)
        regions     = _build_regions(bboxes, text_mask, orig_w, orig_h)

        return CTDResult(text_regions=regions, bubble_mask=bubble_mask)


# ─── Pre/post processing ───────────────────────────────────────────────────

def _preprocess(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    orig_h, orig_w = image.shape[:2]
    if orig_w >= orig_h:
        rw, rh = _DETECT_SIZE, _DETECT_SIZE * orig_h // orig_w
    else:
        rw, rh = _DETECT_SIZE * orig_w // orig_h, _DETECT_SIZE
    resized = cv2.resize(image, (rw, rh)).astype(np.float32) / 255.0
    canvas  = np.zeros((_DETECT_SIZE, _DETECT_SIZE, 3), dtype=np.float32)
    canvas[:rh, :rw] = resized
    return canvas.transpose(2, 0, 1)[None], rw, rh   # (1, 3, 1024, 1024)


def _decode_yolo(
    preds: np.ndarray,
    orig_w: int, orig_h: int,
    rw: int, rh: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Decode blk output → list of (bbox_xyxy, confidence) for text class."""
    w_ratio = orig_w / rw
    h_ratio = orig_h / rh
    boxes: list[tuple[tuple[int,int,int,int], float]] = []

    for p in preds:
        conf = float(p[4] * p[5 + _CLS_TEXT])
        if conf < _CONF_THRESHOLD:
            continue
        cx, cy, bw, bh = (
            p[0] * w_ratio, p[1] * h_ratio,
            p[2] * w_ratio, p[3] * h_ratio,
        )
        x1 = max(0, int(cx - bw / 2) - int(_BBOX_DILATION))
        y1 = max(0, int(cy - bh / 2) - int(_BBOX_DILATION))
        x2 = min(orig_w, int(cx + bw / 2) + int(_BBOX_DILATION))
        y2 = min(orig_h, int(cy + bh / 2) + int(_BBOX_DILATION))
        if x2 > x1 and y2 > y1:
            boxes.append(((x1, y1, x2, y2), conf))

    return _nms(boxes)


def _nms(
    boxes: list[tuple[tuple[int,int,int,int], float]],
) -> list[tuple[tuple[int,int,int,int], float]]:
    boxes = sorted(boxes, key=lambda x: -x[1])
    kept: list[tuple[tuple[int,int,int,int], float]] = []
    for b in boxes:
        if all(_iou(b[0], k[0]) < _NMS_THRESHOLD for k in kept):
            kept.append(b)
    return kept


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, aa + ab - inter)


def _decode_seg(
    seg: np.ndarray,
    orig_w: int, orig_h: int,
    rw: int, rh: int,
) -> np.ndarray:
    """UNet output → uint8 bubble mask (H, W)."""
    crop = seg[:rh, :rw]
    full = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (full > 0.5).astype(np.uint8) * 255


def _decode_det_fused(
    det: np.ndarray,
    seg: np.ndarray,
    orig_w: int, orig_h: int,
    rw: int, rh: int,
) -> np.ndarray:
    """DBNet + UNet fused → binary text pixel mask (H, W) uint8."""
    shrink = det[0]; thresh = det[1]
    h = min(shrink.shape[0], seg.shape[0])
    w = min(shrink.shape[1], seg.shape[1])
    prob  = 1.0 / (1.0 + np.exp(-_DBNET_K * (shrink[:h, :w] - thresh[:h, :w])))
    fused = np.maximum(prob, seg[:h, :w])
    crop  = fused[:rh, :rw]
    full  = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary = (full > _BINARY_THRESHOLD).astype(np.uint8)

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (_HOLE_CLOSE_RADIUS * 2 + 1,) * 2)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (_DILATION_RADIUS  * 2 + 1,) * 2)
    closed  = cv2.morphologyEx(binary * 255, cv2.MORPH_CLOSE, k_close)
    dilated = cv2.dilate(closed, k_dilate)
    return dilated   # uint8 0/255


def _build_regions(
    bboxes: list[tuple[tuple[int,int,int,int], float]],
    text_mask: np.ndarray,
    orig_w: int, orig_h: int,
) -> list[TextRegion]:
    regions: list[TextRegion] = []
    for (x1, y1, x2, y2), conf in bboxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(orig_w, x2), min(orig_h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop    = text_mask[y1c:y2c, x1c:x2c]
        polygon = _mask_to_polygon(crop, x1c, y1c)
        regions.append(TextRegion(
            bbox=(x1c, y1c, x2c, y2c),
            polygon=polygon,
            confidence=conf,
            text_mask=crop,
            mask_x=x1c,
            mask_y=y1c,
        ))
    return regions


def _mask_to_polygon(
    mask: np.ndarray, ox: int, oy: int,
) -> tuple[tuple[float, float], ...]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (
            (float(ox),     float(oy)),
            (float(ox + w), float(oy)),
            (float(ox + w), float(oy + h)),
            (float(ox),     float(oy + h)),
        )
    largest = max(contours, key=cv2.contourArea)
    rect    = cv2.minAreaRect(largest)
    box     = cv2.boxPoints(rect)
    return tuple((float(p[0] + ox), float(p[1] + oy)) for p in box)
