"""Scan → InpaintPlan builder (Python side).

Calls LensBlocksDetector + LensNativeGrouper from the typoon-vision wheel,
applies PixelSegStrategy (Otsu per-word-bbox + morphological close) to
produce per-group tight ink masks, encodes as msgpack.

Mask strategy (mirrors packages/typoon-stages/typoon/stages/scan.py):
  burst/SFX          → ObbPerLineStrategy  (per-line OBB tight stripe)
  dialogue + image   → PixelSegStrategy    (word pixels + morph close)
  polygon_fallback   → no pixel data, Rust regen via adaptive lum threshold
"""
from __future__ import annotations

import asyncio
import logging
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import msgpack
from PIL import Image

from typoon_inpaint.artifact_sink import ArtifactSink, NullSink
from typoon_inpaint.domain import (
    BlockClass, EraseRaster, GroupMask, InpaintPlan, MaskOrigin, PageKind,
)

log = logging.getLogger(__name__)

_runtime = None


def _get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    from typoon.vision._backends.comic_detr import load_session
    from typoon.vision.detectors.lens.detector import LensBlocksDetector
    from typoon.vision.groupers.lens_native import LensNativeGrouper

    model_path = os.environ.get(
        "COMIC_DETR_MODEL",
        str(Path(__file__).parents[3] / "workers/scan/container/comic-detr-v4s-int8.onnx"),
    )
    comic = load_session(model_path)
    det   = LensBlocksDetector(
        comic_detr=comic,
        endpoint=os.environ.get("LENS_ENDPOINT") or None,
        max_concurrent=10,
    )
    grp = LensNativeGrouper()

    class _R:
        detector = det
        grouper  = grp
    _runtime = _R()
    return _runtime


async def build_plan_for_image(
    image_path: Path,
    *,
    lang:  str = "ja",
    sink:  ArtifactSink | None = None,
) -> bytes:
    """Detect + group + derive plan → return msgpack bytes."""
    sink = sink or NullSink()
    rt   = _get_runtime()
    img  = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    detection = await rt.detector.detect(img, lang or None)
    groups    = await rt.grouper.group(img, detection, lang or None)

    page_kind  = _detect_page_kind(img)
    blocks     = list(detection.blocks)   # TextBlock[] — needed by pixel_seg

    plan = InpaintPlan(
        page_index=0,
        page_size=(W, H),
        page_kind=page_kind,
        groups=tuple(
            _to_group_mask(i, g, blocks, img)
            for i, g in enumerate(groups)
        ),
    )

    plan_bytes = _encode_plan(plan)
    sink.write("plan.msgpack", plan_bytes)
    sink.write("groups.json", _groups_json(plan))
    log.info("build_plan: %d groups, page_kind=%s", len(groups), page_kind)
    return plan_bytes


# ── Mask derivation ────────────────────────────────────────────────────────

def _derive_origin(g) -> MaskOrigin:
    if g.source == "ctd" and (g.erase_masks or ()):
        return "ctd_unet"
    if g.used_fallback or not (g.erase_masks or ()):
        return "polygon_fallback"
    return "lens_obb" if abs(g.rotation_deg) > 1.0 else "lens_aabb"


def _classify(g) -> BlockClass:
    from typoon.vision.groupers._classify import classify_block
    class _B:
        rotation_deg = g.rotation_deg
        bbox         = g.bbox
        text         = g.text
    try:
        return classify_block(_B(), g.text or "")
    except Exception:
        return "dialogue"


def _find_members(blocks: list, bbox: tuple) -> tuple:
    """TextBlocks whose centre lies inside group bbox."""
    x1, y1, x2, y2 = bbox
    return tuple(
        b for b in blocks
        if x1 <= (b.bbox[0] + b.bbox[2]) / 2 <= x2
        and y1 <= (b.bbox[1] + b.bbox[3]) / 2 <= y2
    )


def _to_group_mask(
    idx:    int,
    g,                      # BubbleGroup from vision wheel
    blocks: list,           # TextBlock[] from DetectionResult
    img:    np.ndarray,     # full page RGB H×W×3
) -> GroupMask:
    origin = _derive_origin(g)
    class_ = _classify(g)
    poly   = [[float(x), float(y)] for x, y in g.polygon]
    xs     = [p[0] for p in poly]; ys = [p[1] for p in poly]
    bbox   = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    rasters: tuple = ()

    if origin == "polygon_fallback":
        # No pixel data — Rust will regen via adaptive lum threshold
        pass

    elif origin == "ctd_unet":
        # CTD UNet already has per-pixel mask — ship as-is
        rasters = _rasters_from_erase_masks(g.erase_masks or ())

    else:
        # lens_obb / lens_aabb:
        # Strategy (mirrors old scan.py _attach_masks logic):
        #   burst/SFX → OBB per-line stripes (tight, per-line filled)
        #   dialogue  → PixelSeg (Otsu per word bbox + morph close)
        members = _find_members(blocks, bbox)
        rasters = _build_erase_raster(g, members, img, bbox)

        if not rasters:
            # No members / pixel seg produced nothing → fallback
            origin = "polygon_fallback"

    return GroupMask(
        idx=idx, bbox=bbox,
        origin=origin, class_=class_,
        shape_kind=g.shape_kind,
        polygons=(),
        rasters=rasters,
    )


# ── PixelSegStrategy (ported from packages/typoon-vision) ─────────────────

def _build_erase_raster(
    group,
    members: tuple,
    img:     np.ndarray,
    bbox:    tuple[int, int, int, int],
) -> tuple[EraseRaster, ...]:
    """Erase mask cho inpaint container.

    Strategy:
      burst/SFX  → OBB per-line (tight, không flood art/screentone)
      dialogue/narration → word_union AABB + pad (solid rect)

    Insight: mask không cần per-glyph tight.
    Completeness > tightness.
    flat_fill router detect white bg → solid color fill.
    AOT chỉ fire khi bg phức tạp (screentone, art).
    """
    if group.shape_kind == "burst":
        return _obb_per_line_rasters(group, members)

    # Dialogue / narration: word_union bbox + padding
    word_boxes = (
        [w.bbox for m in members for w in m.words]
        or [m.bbox for m in members]
        or [bbox]
    )
    H, W  = img.shape[:2]
    glyph = _glyph_size(members)
    pad   = max(4, glyph // 4)

    x1 = max(0, min(b[0] for b in word_boxes) - pad)
    y1 = max(0, min(b[1] for b in word_boxes) - pad)
    x2 = min(W, max(b[2] for b in word_boxes) + pad)
    y2 = min(H, max(b[3] for b in word_boxes) + pad)
    if x2 <= x1 or y2 <= y1:
        return ()

    data = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    return (_make_raster(x1, y1, data),)


def _glyph_size(members: tuple) -> int:
    try:
        from typoon.vision.groupers._spatial_join import _median_glyph_size
        return _median_glyph_size(list(members)) if members else 10
    except Exception:
        return 10


# ── ObbPerLineStrategy (ported from packages/typoon-vision) ───────────────

def _obb_per_line_rasters(group, members: tuple) -> tuple[EraseRaster, ...]:
    """Per-line OBB filled polygons for burst/SFX groups."""
    from typoon.vision.groupers._spatial_join import (
        _contains_center, _is_column_layout, _line_anchored_obb,
        _MASK_PAD_FACTOR, _MASK_PAD_MIN_PX, _median_glyph_size,
    )

    shape_kind = group.shape_kind
    glyph      = _median_glyph_size(list(members)) if members else 10
    factor     = _MASK_PAD_FACTOR.get(shape_kind, _MASK_PAD_FACTOR["dialogue"])
    pad        = max(_MASK_PAD_MIN_PX, int(glyph * factor))

    rasters: list[EraseRaster] = []

    for m in members:
        if _is_column_layout([m]):
            x1, y1, x2, y2 = m.bbox
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            img = np.full((h, w), 255, dtype=np.uint8)
            rasters.append(_make_raster(x1, y1, img))
            continue

        if not m.lines:
            x1, y1, x2, y2 = m.bbox
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            img = np.full((h, w), 255, dtype=np.uint8)
            rasters.append(_make_raster(x1, y1, img))
            continue

        for line in m.lines:
            words_in_line = [
                w.bbox for w in m.words if _contains_center(line.bbox, w.bbox)
            ]
            obb = _line_anchored_obb(words_in_line, line.bbox, pad) \
                  if len(words_in_line) >= 2 else None

            if obb is not None:
                pts = obb
                pt_xs = [p[0] for p in pts]; pt_ys = [p[1] for p in pts]
                ox1, oy1 = int(min(pt_xs)), int(min(pt_ys))
                ox2, oy2 = int(max(pt_xs)) + 1, int(max(pt_ys)) + 1
                w, h = max(1, ox2 - ox1), max(1, oy2 - oy1)
                img = np.zeros((h, w), dtype=np.uint8)
                local = np.array(
                    [[int(p[0]) - ox1, int(p[1]) - oy1] for p in pts],
                    dtype=np.int32,
                )
                cv2.fillPoly(img, [local], 255)
                rasters.append(_make_raster(ox1, oy1, img))
            else:
                x1, y1, x2, y2 = line.bbox
                x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                img = np.full((h, w), 255, dtype=np.uint8)
                rasters.append(_make_raster(x1, y1, img))

    return tuple(rasters)


def _rasters_from_erase_masks(erase_masks) -> tuple[EraseRaster, ...]:
    """CTD path: ship em.image bytes directly."""
    return tuple(
        _make_raster(int(em.x), int(em.y), em.image.astype(np.uint8))
        for em in erase_masks
    )


def _make_raster(x: int, y: int, img: np.ndarray) -> EraseRaster:
    """Pack a 2D uint8 mask into zlib-compressed EraseRaster."""
    h, w = img.shape[:2]
    return EraseRaster(
        x=x, y=y, w=w, h=h,
        data=zlib.compress(img.tobytes(), level=1),
    )


# ── Page kind ─────────────────────────────────────────────────────────────

def _detect_page_kind(img: np.ndarray) -> PageKind:
    H, W = img.shape[:2]
    if H / W > 2.5:
        return "webtoon"
    lab    = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.int16) - 128
    chroma = float(np.mean(np.abs(lab)))
    return "bw" if chroma < 6.0 else "color"


# ── Msgpack encoder ────────────────────────────────────────────────────────

def _encode_plan(plan: InpaintPlan) -> bytes:
    def _enc_group(g: GroupMask) -> dict:
        return {
            "idx":        g.idx,
            "bbox":       list(g.bbox),
            "origin":     g.origin,
            "class":      g.class_,
            "shape_kind": g.shape_kind,
            "polygons":   [],
            "rasters":    [
                {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "data": r.data}
                for r in g.rasters
            ],
        }

    payload = {
        "page_index": plan.page_index,
        "page_size":  list(plan.page_size),
        "page_kind":  plan.page_kind,
        "groups":     [_enc_group(g) for g in plan.groups],
    }
    return msgpack.packb(payload, use_bin_type=True)


def _groups_json(plan: InpaintPlan) -> str:
    import json
    return json.dumps([
        {
            "idx": g.idx, "bbox": list(g.bbox),
            "origin": g.origin, "class": g.class_,
            "shape_kind": g.shape_kind,
            "n_rasters": len(g.rasters),
        }
        for g in plan.groups
    ], indent=2)
