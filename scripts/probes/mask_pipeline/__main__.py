"""Repro full scan→mask pipeline on a single page, offline.

Mirrors workers/scan/container/main.py (detect + group + _build_mask)
plus the Rust close_mask_per_block + detect_strokes_in_bbox so we can
inspect every intermediate without deploying.

Usage:
    python -m scripts.probes.mask_pipeline <image>
        [--out debug-runs/mask-probe-<name>]
        [--lang ja]
        [--no-lens]    # skip real Lens (offline-only, polygon=bbox)

Outputs:
    <out>/
      00_input.png
      01_groups.json
      02_erase_only.png
      03_polygon_fallback.png
      04_after_scan_dilate.png
      05_density_per_group.json
      06_close_per_block.png
      07_stroke_detect.png
      08_overlay.png
      manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "workers/scan/container/comic-detr-v4s-int8.onnx"

logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")
log = logging.getLogger("mask_pipeline")


# ─── Replay of workers/scan/container/main.py:_build_mask (verbatim) ─────────

def build_mask_scan(groups, W: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (erase_only, polygon_fallback, after_scan_dilate)."""
    erase_only       = np.zeros((H, W), dtype=np.uint8)
    polygon_fallback = np.zeros((H, W), dtype=np.uint8)
    for g in groups:
        used_erase = False
        for em in getattr(g, "erase_masks", ()) or ():
            ex, ey = int(em.x), int(em.y)
            tile = em.image
            th, tw = tile.shape[:2]
            x0 = max(0, ex); y0 = max(0, ey)
            x1 = min(W, ex + tw); y1 = min(H, ey + th)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0 = x0 - ex; sy0 = y0 - ey
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
            region = erase_only[y0:y1, x0:x1]
            np.maximum(region, tile[sy0:sy1, sx0:sx1], out=region)
            used_erase = True
        if not used_erase:
            poly = np.array(g.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(polygon_fallback, [poly], 255)

    page_mask = np.maximum(erase_only, polygon_fallback)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    after_scan_dilate = cv2.dilate(page_mask, kernel, iterations=1)
    return erase_only, polygon_fallback, after_scan_dilate


# ─── Port of crates/inpaint/src/page.rs:close_mask_per_block (Python) ────────

CLOSE_RADIUS_MIN = 2


def close_radius_frac(class_name: str | None) -> float:
    return {"sfx": 0.15, "narration": 0.12}.get(class_name or "", 0.10)


def close_radius_for(bbox, class_name) -> int:
    x1, y1, x2, y2 = bbox
    short = max(0, min(x2 - x1, y2 - y1))
    frac = close_radius_frac(class_name)
    return max(CLOSE_RADIUS_MIN, round(short * frac))


def dilate_rect_bin(src: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return src.copy()
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(src, kernel, iterations=1)


def erode_rect_bin(src: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return src.copy()
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(src, kernel, iterations=1)


def fill_enclosed_holes_bin(mask: np.ndarray) -> np.ndarray:
    """BFS from border zeros; pixels=0 unreachable from border → flip to 1."""
    h, w = mask.shape
    bin_ = (mask > 0).astype(np.uint8)
    inv = (1 - bin_).astype(np.uint8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    outside = (flood[1:-1, 1:-1] == 2).astype(np.uint8)
    holes = ((bin_ == 0) & (outside == 0)).astype(np.uint8)
    return ((bin_ | holes) > 0).astype(np.uint8)


def detect_strokes_in_bbox(img: np.ndarray, bbox, pad: int) -> np.ndarray:
    """Port of Rust detect_strokes_in_bbox. img is HxWx3 uint8 RGB.

    Two rules:
      - (R - B) > 20 AND sat > 30 AND lum < 160  → coloured ink
      - lum < 80                                  → dark ink
    """
    H, W = img.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)
    bx1, by1, bx2, by2 = bbox
    x0 = max(0, bx1 - pad); y0 = max(0, by1 - pad)
    x1 = min(W - 1, bx2 + pad); y1 = min(H - 1, by2 + pad)
    if x1 <= x0 or y1 <= y0:
        return out
    crop = img[y0:y1 + 1, x0:x1 + 1].astype(np.int16)
    r = crop[..., 0]; g = crop[..., 1]; b = crop[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.int16)
    coloured = ((r - b) > 20) & (sat > 30) & (lum < 160)
    dark = lum < 80
    out[y0:y1 + 1, x0:x1 + 1] = ((coloured | dark)).astype(np.uint8)
    return out


def close_mask_per_block(
    page_mask: np.ndarray,
    groups: list[dict],
    img: np.ndarray,
    density_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (closed_mask_255, stroke_overlay_255, per_group_diagnostics)."""
    H, W = page_mask.shape
    bin_ = (page_mask >= 127).astype(np.uint8)
    result = bin_.copy()
    stroke_overlay = np.zeros((H, W), dtype=np.uint8)
    diagnostics = []

    for g in groups:
        bx1, by1, bx2, by2 = g["bbox"]
        class_name = g.get("class")
        r = close_radius_for(g["bbox"], class_name)
        px0 = max(0, bx1 - r); py0 = max(0, by1 - r)
        px1 = min(W, bx2 + r); py1 = min(H, by2 + r)
        if px1 <= px0 or py1 <= py0:
            continue

        # bbox-only density on raw bin
        cbx1 = max(0, bx1); cby1 = max(0, by1)
        cbx2 = min(W, bx2); cby2 = min(H, by2)
        bbox_area = max(0, cbx2 - cbx1) * max(0, cby2 - cby1)
        bbox_mask_on = int(bin_[cby1:cby2, cbx1:cbx2].sum()) if bbox_area else 0
        density = bbox_mask_on / bbox_area if bbox_area else 0.0

        patch = bin_[py0:py1, px0:px1].copy()
        used_stroke = False
        if density > density_threshold:
            strokes = detect_strokes_in_bbox(img, g["bbox"], r)
            patch = strokes[py0:py1, px0:px1].copy()
            stroke_overlay[py0:py1, px0:px1] = np.maximum(
                stroke_overlay[py0:py1, px0:px1], strokes[py0:py1, px0:px1] * 255
            )
            used_stroke = True

        dilated = dilate_rect_bin(patch, r)
        closed = erode_rect_bin(dilated, r)
        closed = fill_enclosed_holes_bin(closed * 255)

        # outsider guard
        bridges = any(
            (o is not g)
            and px0 <= (o["bbox"][0] + o["bbox"][2]) / 2 < px1
            and py0 <= (o["bbox"][1] + o["bbox"][3]) / 2 < py1
            for o in groups
        )

        block_orig = bin_[py0:py1, px0:px1]
        for y in range(py1 - py0):
            for x in range(px1 - px0):
                if closed[y, x] == 0:
                    continue
                if block_orig[y, x] == 0 and bridges:
                    continue
                result[py0 + y, px0 + x] = 1

        diagnostics.append({
            "idx": g["idx"],
            "bbox": g["bbox"],
            "class": class_name,
            "shape_kind": g.get("shape_kind"),
            "close_radius": r,
            "bbox_area": bbox_area,
            "bbox_mask_on": bbox_mask_on,
            "density": round(density, 4),
            "stroke_regen": used_stroke,
            "bridges": bridges,
        })

    return (result * 255).astype(np.uint8), stroke_overlay, diagnostics


# ─── Overlay viz ─────────────────────────────────────────────────────────────

def overlay_groups(img_rgb: np.ndarray, closed_mask: np.ndarray, diagnostics: list[dict]) -> np.ndarray:
    base = img_rgb.copy()
    red = np.zeros_like(base); red[..., 0] = 255
    a = (closed_mask >= 127).astype(np.float32)[..., None] * 0.45
    base = (base * (1 - a) + red * a).astype(np.uint8)

    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()
    for d in diagnostics:
        x1, y1, x2, y2 = d["bbox"]
        col = (255, 255, 0) if d["stroke_regen"] else (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        tag = f"#{d['idx']} d={d['density']:.2f} r={d['close_radius']}"
        if d["stroke_regen"]:
            tag += " STROKE"
        if d["bridges"]:
            tag += " BRIDGE"
        draw.text((x1 + 2, max(0, y1 - 14)), tag, fill=col, font=font)
    return np.array(pil)


# ─── Pipeline ────────────────────────────────────────────────────────────────

def to_jsonable(obj):
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def serialise_group(idx: int, g) -> dict:
    poly = [[float(x), float(y)] for x, y in g.polygon]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return {
        "idx": idx,
        "bbox": bbox,
        "polygon": poly,
        "source_text": g.text,
        "confidence": float(g.confidence),
        "shape_kind": g.shape_kind,
        "used_fallback": bool(g.used_fallback),
        "rotation_deg": float(g.rotation_deg),
        "text_direction": g.text_direction,
        "n_erase_masks": len(g.erase_masks or ()),
        "n_text_masks": len(g.text_masks or ()),
        # NOTE: `class` is what Rust expects (sfx/dialogue/narration);
        # vision wheel doesn't ship it on BubbleGroup → reclassify here.
        "class": classify_for_rust(g),
    }


def classify_for_rust(g) -> str:
    """Mirror _classify.classify_block roughly: rotation/aspect/char rules."""
    from typoon.vision.groupers._classify import classify_block
    from typoon.vision.contracts import TextBlock
    # Build a minimal TextBlock-like — classify_block only reads
    # rotation_deg, aspect via bbox, text length.
    class _B:
        rotation_deg = g.rotation_deg
        bbox = g.bbox
        text = g.text
    try:
        return classify_block(_B(), g.text or "")
    except Exception:
        return "dialogue"


async def run(image_path: Path, out_dir: Path, lang: str, use_lens: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("input: %s → %s", image_path, out_dir)

    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    Image.fromarray(img).save(out_dir / "00_input.png")

    if not use_lens:
        log.warning("Running with --no-lens: empty groups, sanity only")
        groups = ()
    else:
        from typoon.vision.detectors.lens.detector import LensBlocksDetector
        from typoon.vision.groupers.lens_native import LensNativeGrouper
        from typoon.vision._backends.comic_detr import load_session

        t0 = time.perf_counter()
        comic = load_session(str(MODEL_PATH))
        detector = LensBlocksDetector(
            comic_detr=comic,
            endpoint=os.environ.get("LENS_ENDPOINT") or None,
            max_concurrent=10,
        )
        grouper = LensNativeGrouper()
        log.info("runtime ready %.1fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        detection = await detector.detect(img, lang or None)
        log.info("detect %.1fs  blocks=%d", time.perf_counter() - t0, len(detection.blocks))

        t0 = time.perf_counter()
        groups = await grouper.group(img, detection, lang or None)
        log.info("group  %.1fs  groups=%d", time.perf_counter() - t0, len(groups))

    serial = [serialise_group(i, g) for i, g in enumerate(groups)]
    (out_dir / "01_groups.json").write_text(json.dumps(to_jsonable(serial), indent=2))

    erase_only, polygon_fallback, after_scan = build_mask_scan(groups, W, H)
    Image.fromarray(erase_only).save(out_dir / "02_erase_only.png")
    Image.fromarray(polygon_fallback).save(out_dir / "03_polygon_fallback.png")
    Image.fromarray(after_scan).save(out_dir / "04_after_scan_dilate.png")

    closed, stroke_overlay, diagnostics = close_mask_per_block(after_scan, serial, img)
    Image.fromarray(closed).save(out_dir / "06_close_per_block.png")
    Image.fromarray(stroke_overlay).save(out_dir / "07_stroke_detect.png")
    (out_dir / "05_density_per_group.json").write_text(json.dumps(diagnostics, indent=2))

    over = overlay_groups(img, closed, diagnostics)
    Image.fromarray(over).save(out_dir / "08_overlay.png")

    n_fallback = sum(1 for g in groups if not (g.erase_masks or ()))
    n_stroke   = sum(1 for d in diagnostics if d["stroke_regen"])
    n_bridge   = sum(1 for d in diagnostics if d["bridges"])
    summary = {
        "image":               str(image_path),
        "size":                [W, H],
        "n_groups":            len(groups),
        "n_polygon_fallback":  n_fallback,
        "n_stroke_regen":      n_stroke,
        "n_bridge":            n_bridge,
        "density_mean":        round(float(np.mean([d["density"] for d in diagnostics] or [0])), 3),
        "density_max":         round(float(max((d["density"] for d in diagnostics), default=0)), 3),
        "density_p85":         round(float(np.percentile([d["density"] for d in diagnostics] or [0], 85)), 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))

    log.info("DONE  groups=%d  fallback=%d  stroke_regen=%d  bridges=%d",
             len(groups), n_fallback, n_stroke, n_bridge)
    log.info("artifacts: %s", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--no-lens", action="store_true")
    args = ap.parse_args()

    out = args.out or (ROOT / "debug-runs" / f"mask-probe-{args.image.stem}")
    asyncio.run(run(args.image, out, args.lang, use_lens=not args.no_lens))


if __name__ == "__main__":
    main()
