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
    BlockClass, Coarse, EraseRaster, Group, InpaintPlan, MaskKind, PageKind,
    PLAN_VERSION, Precise, Regen, dilate_for_glyph,
)

log = logging.getLogger(__name__)

_runtime = None
_ctd_sess = None

CTD_ONNX_PATH = os.environ.get(
    "CTD_ONNX_PATH",
    str(Path.home() / ".cache/huggingface/hub/models--mayocream--comic-text-detector-onnx/"
         "snapshots/a5d67ec772adef819ef5b0e7aa701fcf4c8bf74a/comic-text-detector.onnx"),
)


def _get_ctd_sess():
    """Lazy-load CTD ONNX session for per-pixel text mask inference."""
    global _ctd_sess
    if _ctd_sess is None:
        import onnxruntime as ort
        _ctd_sess = ort.InferenceSession(
            CTD_ONNX_PATH,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        log.info("CTD ONNX loaded: %s", CTD_ONNX_PATH)
    return _ctd_sess


def _ctd_text_mask(img: np.ndarray) -> np.ndarray:
    """Run CTD UNet seg-only → bubble interior mask.

    Mirrors packages/typoon-vision/typoon/vision/masks/ctd_seg_runner.py
    (commit f5de54b — production-tested):

      1. Run ONNX, lấy CHỈ seg output (~265ms vs 308ms cho full 3 outputs)
      2. resize seg → orig size (INTER_LINEAR preserves soft boundary)
      3. binary = seg > 0.5
      4. morph close r=10  ← fill holes inside bubble outline
      5. dilate r=3        ← expand cover balloon edge stroke

    Output uint8 (H, W) 0/255 = bubble interior mask. CTD bổ sung cho
    case Lens word boxes yếu (JA/ZH), KHÔNG phải ink-perfect mask.
    """
    HOLE_CLOSE_RADIUS = 10
    DILATION_RADIUS   = 6
    DSZ               = 1024

    sess = _get_ctd_sess()
    H, W = img.shape[:2]
    if W >= H:
        rw, rh = DSZ, DSZ * H // W
    else:
        rw, rh = DSZ * W // H, DSZ
    resized = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
    canvas  = np.zeros((DSZ, DSZ, 3), dtype=np.float32)
    canvas[:rh, :rw] = resized
    inp = canvas.transpose(2, 0, 1)[None]

    # seg-only run (skip blk/det decoding)
    [seg_out] = sess.run(["seg"], {"images": inp})
    seg = seg_out[0, 0]

    crop   = seg[:rh, :rw]
    full   = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    binary = (full > 0.5).astype(np.uint8) * 255

    k_close  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (HOLE_CLOSE_RADIUS * 2 + 1,) * 2
    )
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATION_RADIUS * 2 + 1,) * 2
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    return cv2.dilate(closed, k_dilate)


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
    page_index: int = 0,
    lang:  str = "ja",
    sink:  ArtifactSink | None = None,
) -> bytes:
    """Detect groups + CTD text_mask → per-group ink mask → plan msgpack.

    Pipeline:
      Lens detect      → BubbleGroup[] with bbox
      CTD UNet 1 pass  → per-pixel text_mask cho toàn page
      per group: crop text_mask[bbox] → EraseRaster

    page_index == 0 + signals khác → skip inpaint (poster/cover detection).
    """
    sink = sink or NullSink()
    rt   = _get_runtime()
    img  = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    detection = await rt.detector.detect(img, lang or None)
    raw_groups = await rt.grouper.group(img, detection, lang or None)
    blocks    = list(detection.blocks)

    # Filter noise groups (watermarks, tiny fragments, oversized SFX,
    # rotated single-word artifacts).
    groups = _filter_noise(raw_groups)
    n_rej  = len(raw_groups) - len(groups)
    if n_rej:
        log.info("filter: rejected %d/%d noise groups", n_rej, len(raw_groups))

    # Poster/cover detection: skip inpaint hoàn toàn cho poster pages.
    # Dùng signals nội tại của page, không dùng page_index.
    if _is_poster(groups, W, H):
        log.info("poster detected (page %d) — skip inpaint", page_index)
        plan = InpaintPlan(
            page_index=page_index,
            page_size=(W, H),
            page_kind=_detect_page_kind(img),
            groups=(),    # empty → inpaint container passthrough
        )
        plan_bytes = _encode_plan(plan)
        sink.write("plan.msgpack", plan_bytes)
        sink.write("groups.json", '[]  # poster — skipped')
        return plan_bytes

    # CTD UNet seg-only mask — chỉ chạy cho JA/ZH (CTD train trên ja manga,
    # word boxes của Lens yếu trên tategaki / nét chữ JA/ZH).
    # Non-JA hoặc SKIP_CTD=1 → bỏ qua, fallback PixelSeg (Otsu per word).
    text_mask: np.ndarray | None = None
    skip_ctd = os.environ.get("SKIP_CTD", "0") == "1"
    if not skip_ctd and lang and lang.lower()[:2] in ("ja", "zh"):
        try:
            text_mask = await asyncio.to_thread(_ctd_text_mask, img)
        except Exception as e:
            log.warning("CTD UNet failed (%s), fallback to PixelSeg", e)
            text_mask = None

    page_kind = _detect_page_kind(img)

    plan = InpaintPlan(
        page_index=page_index,
        page_size=(W, H),
        page_kind=page_kind,
        groups=tuple(
            _build_group(i, g, blocks, img, text_mask)
            for i, g in enumerate(groups)
        ),
    )

    plan_bytes = _encode_plan(plan)
    sink.write("plan.msgpack", plan_bytes)
    sink.write("groups.json", _groups_json(plan))
    log.info("build_plan: %d groups, page_kind=%s", len(groups), page_kind)
    return plan_bytes


# ── Mask derivation ────────────────────────────────────────────────────────

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


def _build_group(
    idx:       int,
    g,                                  # BubbleGroup from vision wheel
    blocks:    list,                    # TextBlock[] from DetectionResult
    img:       np.ndarray,              # full page RGB
    text_mask: np.ndarray | None,       # CTD UNet bubble mask (None for non-JA)
) -> Group:
    """Pick a `MaskKind` for this group. Vision owns the policy.

    Strategy:
      1. burst/SFX                      → Coarse (OBB per-line stripe)
      2. dialogue + text_mask available → Precise (CTD UNet crop)
      3. dialogue, no text_mask         → Precise (Otsu per-word + morph)
      4. nothing usable                 → Regen (Rust stroke regen from page)
    """
    class_  = _classify(g)
    bbox    = (int(g.bbox[0]), int(g.bbox[1]), int(g.bbox[2]), int(g.bbox[3]))
    members = _find_members(blocks, bbox)
    glyph   = _glyph_size(members) if members else 10

    mask: MaskKind | None = None

    # 1. burst/SFX → Coarse OBB per-line stripe. Vision baked some
    #    pre-padding inside `_obb_per_line_rasters` already; we still
    #    ask inpaint for a small outward dilate proportional to glyph.
    if g.shape_kind == "burst":
        rasters = _obb_per_line_rasters(g, members, img)
        if rasters:
            raster = _merge_rasters(rasters)
            mask = Coarse(raster=raster, dilate_px=dilate_for_glyph(glyph))

    # 2. CTD UNet available → Precise crop
    if mask is None and text_mask is not None:
        H, W = text_mask.shape[:2]
        x1, y1, x2, y2 = bbox
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(W, x2); y2c = min(H, y2)
        if x2c > x1c and y2c > y1c:
            crop = text_mask[y1c:y2c, x1c:x2c].copy()
            if crop.any():
                mask = Precise(raster=_make_raster(x1c, y1c, crop))

    # 3. PixelSeg Otsu+morph → Precise (output already glyph-aligned)
    if mask is None:
        rasters = _pixel_seg_rasters(members, img, bbox)
        if rasters:
            mask = Precise(raster=_merge_rasters(rasters))

    # 4. Nothing usable → ask inpaint to regen from page pixels
    if mask is None:
        mask = Regen()

    return Group(
        idx=idx, bbox=bbox, class_=class_,
        shape_kind=g.shape_kind, mask=mask,
    )


def _merge_rasters(rasters: tuple[EraseRaster, ...]) -> EraseRaster:
    """OR a tuple of rasters into a single raster covering their union.

    Contract requires exactly one raster per Precise/Coarse variant —
    vision-side merge keeps the wire simple and pushes per-stripe
    bookkeeping out of inpaint.
    """
    if len(rasters) == 1:
        return rasters[0]
    # Union bbox in page coords
    x0 = min(r.x for r in rasters)
    y0 = min(r.y for r in rasters)
    x1 = max(r.x + r.w for r in rasters)
    y1 = max(r.y + r.h for r in rasters)
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for r in rasters:
        pixels = np.frombuffer(
            zlib.decompress(r.data), dtype=np.uint8
        ).reshape(r.h, r.w)
        dy = r.y - y0
        dx = r.x - x0
        np.maximum(canvas[dy:dy + r.h, dx:dx + r.w], pixels,
                   out=canvas[dy:dy + r.h, dx:dx + r.w])
    return _make_raster(x0, y0, canvas)


# ── PixelSegStrategy (ported from f5de54b vision/masks/pixel_seg.py) ──────

_CLIP_PAD = 4   # px margin around word_union for dilation room


def _pixel_seg_rasters(
    members: tuple,
    img:     np.ndarray,
    bbox:    tuple[int, int, int, int],
) -> tuple[EraseRaster, ...]:
    """Otsu per word bbox + 3-step morph close + largest CC.

    Direct port of PixelSegStrategy._pixel_seg from f5de54b.
    Returns 1 raster covering word_union region.
    """
    word_boxes = (
        [w.bbox for m in members for w in m.words]
        or [m.bbox for m in members]
        or [bbox]
    )
    wu_x1 = min(b[0] for b in word_boxes)
    wu_y1 = min(b[1] for b in word_boxes)
    wu_x2 = max(b[2] for b in word_boxes)
    wu_y2 = max(b[3] for b in word_boxes)

    glyph  = _glyph_size(members)
    bridge = max(glyph, 8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pH, pW = gray.shape

    # Seed: Otsu per word bbox
    seed = np.zeros((pH, pW), dtype=np.uint8)
    for wb in word_boxes:
        wx1, wy1 = max(0, wb[0]), max(0, wb[1])
        wx2, wy2 = min(pW, wb[2]), min(pH, wb[3])
        patch = gray[wy1:wy2, wx1:wx2]
        if patch.size < 9:
            continue
        _, bp = cv2.threshold(
            patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        seed[wy1:wy2, wx1:wx2] = bp

    # Crop to word_union + CLIP_PAD
    rx1 = max(0, wu_x1 - _CLIP_PAD)
    ry1 = max(0, wu_y1 - _CLIP_PAD)
    rx2 = min(pW, wu_x2 + _CLIP_PAD)
    ry2 = min(pH, wu_y2 + _CLIP_PAD)
    blob = seed[ry1:ry2, rx1:rx2].copy()
    if not blob.any():
        return ()

    roi_h, roi_w = blob.shape[:2]
    cap = max(2, min(roi_w // 3, roi_h // 3))

    def _r(base: int) -> int:
        return min(base, cap)

    # 3-step morphological close
    for r in (
        _r(max(2, glyph // 4)),
        _r(max(4, int(glyph * 0.8))),
        _r(max(6, int(bridge * 0.7))),
    ):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k)

    # Hard-clip to word_union
    cm = np.zeros_like(blob)
    cy1_ = max(0, wu_y1 - ry1)
    cy2_ = min(roi_h, wu_y2 - ry1)
    cx1_ = max(0, wu_x1 - rx1)
    cx2_ = min(roi_w, wu_x2 - rx1)
    cm[cy1_:cy2_, cx1_:cx2_] = 255
    blob = cv2.bitwise_and(blob, cm)

    # Keep largest CC
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = (labels == largest).astype(np.uint8) * 255

    if not blob.any():
        return ()

    # Dilate + smooth (UNet-like soft boundary)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blob  = cv2.dilate(blob, k_dil, iterations=1)
    blob  = cv2.GaussianBlur(blob, (7, 7), sigmaX=2.0)
    _, blob = cv2.threshold(blob, 80, 255, cv2.THRESH_BINARY)

    return (_make_raster(rx1, ry1, blob),)


def _glyph_size(members: tuple) -> int:
    try:
        from typoon.vision.groupers._spatial_join import _median_glyph_size
        return _median_glyph_size(list(members)) if members else 10
    except Exception:
        return 10


# ── ObbPerLineStrategy (ported from packages/typoon-vision) ───────────────

def _obb_per_line_rasters(
    group,
    members: tuple,
    img:     np.ndarray | None = None,   # unused — kept for API stability
) -> tuple[EraseRaster, ...]:
    """Per-line OBB filled polygons + adaptive dilate.

    Solid-fill OBB polygon (production-tested from f5de54b),
    followed by light dilate to cover stroke anti-alias + small bleed.

    Dilate radius adaptive per group shape + glyph size:
      - dialogue/narration: r ≈ glyph * 0.12, factor 1.0
      - burst/SFX        : r ≈ glyph * 0.12, factor 1.5 (thicker outlines)
      - clamped [2, 10]px (avoid merging adjacent glyphs)
    """
    from typoon.vision.groupers._spatial_join import (
        _contains_center, _is_column_layout, _line_anchored_obb,
        _MASK_PAD_FACTOR, _MASK_PAD_MIN_PX, _median_glyph_size,
    )

    shape_kind = group.shape_kind
    glyph      = _median_glyph_size(list(members)) if members else 10
    factor     = _MASK_PAD_FACTOR.get(shape_kind, _MASK_PAD_FACTOR["dialogue"])
    pad        = max(_MASK_PAD_MIN_PX, int(glyph * factor))

    rasters: list[EraseRaster] = []

    def _emit_aabb(x1: int, y1: int, x2: int, y2: int) -> None:
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        data = np.full((h, w), 255, dtype=np.uint8)
        short_edge = min(w, h)
        data = _dilate_obb(data, glyph, short_edge, shape_kind)
        rasters.append(_make_raster(x1, y1, data))

    for m in members:
        if _is_column_layout([m]):
            x1, y1, x2, y2 = m.bbox
            _emit_aabb(x1, y1, x2, y2)
            continue

        if not m.lines:
            x1, y1, x2, y2 = m.bbox
            _emit_aabb(x1, y1, x2, y2)
            continue

        for line in m.lines:
            words_in_line = [
                w.bbox for w in m.words if _contains_center(line.bbox, w.bbox)
            ]
            obb = _line_anchored_obb(words_in_line, line.bbox, pad) \
                  if len(words_in_line) >= 2 else None

            if obb is not None:
                pt_xs = [p[0] for p in obb]; pt_ys = [p[1] for p in obb]
                ox1, oy1 = int(min(pt_xs)), int(min(pt_ys))
                ox2, oy2 = int(max(pt_xs)) + 1, int(max(pt_ys)) + 1
                w, h = max(1, ox2 - ox1), max(1, oy2 - oy1)
                data = np.zeros((h, w), dtype=np.uint8)
                local = np.array(
                    [[int(p[0]) - ox1, int(p[1]) - oy1] for p in obb],
                    dtype=np.int32,
                )
                cv2.fillPoly(data, [local], 255)
                short_edge = min(w, h)
                data = _dilate_obb(data, glyph, short_edge, shape_kind)
                rasters.append(_make_raster(ox1, oy1, data))
            else:
                x1, y1, x2, y2 = line.bbox
                _emit_aabb(x1, y1, x2, y2)

    return tuple(rasters)


def _dilate_obb(
    data: np.ndarray, glyph: int, short_edge: int, shape_kind: str,
) -> np.ndarray:
    """Adaptive dilate radius based on glyph + short_edge + shape kind."""
    if glyph >= 10:
        base = glyph * 0.12
    else:
        base = short_edge * 0.04
    factor = 1.5 if shape_kind == "burst" else 1.0
    r = int(max(2, min(10, base * factor)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    return cv2.dilate(data, k)


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
    payload = {
        "version":    PLAN_VERSION,
        "page_index": plan.page_index,
        "page_size":  list(plan.page_size),
        "page_kind":  plan.page_kind,
        "groups": [
            {
                "idx":        g.idx,
                "bbox":       list(g.bbox),
                "class":      g.class_,
                "shape_kind": g.shape_kind,
                "mask":       g.mask.to_wire(),
            }
            for g in plan.groups
        ],
    }
    return msgpack.packb(payload, use_bin_type=True)


def _groups_json(plan: InpaintPlan) -> str:
    import json
    return json.dumps([
        {
            "idx":        g.idx,
            "bbox":       list(g.bbox),
            "class":      g.class_,
            "shape_kind": g.shape_kind,
            "kind":       g.mask.to_wire()["kind"],
        }
        for g in plan.groups
    ], indent=2)


# ── Noise filter (ported from f5de54b vision/filters/) ────────────────────
#
# Hard-reject groups matching any of:
#   1. Watermark tokens in text (READMANGA, MANGADEX, ...)
#   2. Tiny fragments: area<1500 AND words<=1 AND chars<=2
#   3. Extreme rotation >100° with single word (upside-down OCR artifacts)
#   4. Oversized SFX (burst, area > 2.5× dialogue median) overlapping bubble

_WATERMARK_TOKENS = frozenset({
    "DO NOT MIRROR", "DO NOT REPOST", "MANGA STREAM", "MANGASTREAM",
    "SCANLATION", "SCANS.NET", "READMANGA", "MANGADEX", "BAOZIMH",
    "MANHUAGUI", "COPYMANGA",
})


def _is_watermark(g) -> bool:
    txt = (g.text or "").upper()
    return any(tok in txt for tok in _WATERMARK_TOKENS)


def _is_tiny_fragment(g) -> bool:
    x1, y1, x2, y2 = g.bbox
    area    = (x2 - x1) * (y2 - y1)
    text    = g.text or ""
    n_words = len(text.split())
    n_chars = sum(1 for c in text if not c.isspace())
    return area < 1500 and n_words <= 1 and n_chars <= 2


def _is_rotation_artifact(g) -> bool:
    text    = g.text or ""
    n_words = len(text.split())
    return abs(g.rotation_deg) > 100.0 and n_words <= 1


def _is_oversized_sfx(g, all_groups) -> bool:
    if g.shape_kind != "burst":
        return False
    x1, y1, x2, y2 = g.bbox
    sfx_area = max(1, (x2 - x1) * (y2 - y1))

    body_areas = []
    for o in all_groups:
        if o is g or o.shape_kind != "dialogue":
            continue
        ox1, oy1, ox2, oy2 = o.bbox
        body_areas.append(max(1, (ox2 - ox1) * (oy2 - oy1)))
    if not body_areas:
        return False

    body_areas.sort()
    body_median = body_areas[len(body_areas) // 2]
    if sfx_area / body_median < 2.5:
        return False

    # Must also overlap another group (otherwise just art in empty space)
    for o in all_groups:
        if o is g:
            continue
        ox1, oy1, ox2, oy2 = o.bbox
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if overlap >= 500:
            return True
    return False


def _filter_noise(groups: tuple) -> tuple:
    kept = []
    for g in groups:
        if _is_watermark(g):
            log.debug("filter: watermark %r", (g.text or "")[:30])
            continue
        if _is_tiny_fragment(g):
            log.debug("filter: tiny fragment %r bbox=%s",
                      (g.text or "")[:30], g.bbox)
            continue
        if _is_rotation_artifact(g):
            log.debug("filter: rotation artifact %.1f° %r",
                      g.rotation_deg, (g.text or "")[:30])
            continue
        if _is_oversized_sfx(g, groups):
            log.debug("filter: oversized SFX %r bbox=%s",
                      (g.text or "")[:30], g.bbox)
            continue
        kept.append(g)
    return tuple(kept)


# ── Poster / cover detection ──────────────────────────────────────────────
#
# Heuristic dựa trên signals nội tại của page. Poster/cover khác manga
# narrative page ở chỗ text spread khắp page với mix dialogue/sfx cao.
#
# Page là poster khi:
#   (total_group_area / page_area > 0.20) AND (sfx_ratio > 0.4)
#
# Threshold derived từ chainsaw cover (24.7% area, 57% sfx) vs manga page
# bình thường (~5-10% area, 0-30% sfx).

def _is_poster(groups: tuple, page_w: int, page_h: int) -> bool:
    if not groups:
        return False

    page_area = page_w * page_h
    if page_area == 0:
        return False

    n_total = len(groups)
    n_burst = sum(1 for g in groups if g.shape_kind == "burst")
    sfx_ratio = n_burst / n_total

    total_group_area = 0
    for g in groups:
        x1, y1, x2, y2 = g.bbox
        total_group_area += max(0, x2 - x1) * max(0, y2 - y1)
    area_ratio = total_group_area / page_area

    is_poster = area_ratio > 0.20 and sfx_ratio > 0.4
    if is_poster:
        log.info(
            "poster signals: n=%d sfx_ratio=%.2f area_ratio=%.2f",
            n_total, sfx_ratio, area_ratio,
        )
    return is_poster
