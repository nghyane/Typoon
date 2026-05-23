"""Scan → InpaintPlan builder (Python side).

Calls LensBlocksDetector + LensNativeGrouper from the typoon-vision wheel,
derives mask_origin + class from BubbleGroup fields, encodes as msgpack.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import msgpack
import numpy as np
from PIL import Image

from typoon_inpaint.artifact_sink import ArtifactSink, NullSink
from typoon_inpaint.domain import (
    BlockClass, GroupMask, InpaintPlan, MaskOrigin, PageKind, PROFILES,
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
    grp   = LensNativeGrouper()

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

    page_kind = _detect_page_kind(img)
    plan      = InpaintPlan(
        page_index=0,
        page_size=(W, H),
        page_kind=page_kind,
        groups=tuple(_to_group_mask(i, g) for i, g in enumerate(groups)),
    )

    plan_bytes = _encode_plan(plan)
    sink.write("plan.msgpack", plan_bytes)
    sink.write("groups.json", _groups_json(plan))
    log.info("build_plan: %d groups, page_kind=%s", len(groups), page_kind)
    return plan_bytes


# ── Derivation helpers ────────────────────────────────────────────────────

def _derive_origin(g) -> MaskOrigin:
    if g.source == "ctd" and (g.erase_masks or ()):
        return "ctd_unet"
    if g.used_fallback or not (g.erase_masks or ()):
        return "polygon_fallback"
    from typoon.vision.contracts import TextMask
    # OBB = mask is not purely AABB (has non-zero rotation signal)
    if abs(g.rotation_deg) > 1.0:
        return "lens_obb"
    for em in g.erase_masks:
        h, w = em.image.shape[:2]
        if h != (g.bbox[3] - g.bbox[1]) or w != (g.bbox[2] - g.bbox[0]):
            return "lens_obb"
    return "lens_aabb"


def _classify(g) -> BlockClass:
    from typoon.vision.groupers._classify import classify_block
    class _B:
        rotation_deg = g.rotation_deg
        bbox = g.bbox
        text = g.text
    try:
        return classify_block(_B(), g.text or "")
    except Exception:
        return "dialogue"


def _to_group_mask(idx: int, g) -> GroupMask:
    origin = _derive_origin(g)
    class_ = _classify(g)
    poly   = [[float(x), float(y)] for x, y in g.polygon]
    xs     = [p[0] for p in poly]; ys = [p[1] for p in poly]
    bbox   = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    polygons: tuple = ()
    rasters:  tuple = ()

    if origin == "polygon_fallback":
        # No pixel data — Rust will regen via Canny
        pass

    elif origin == "ctd_unet":
        # CTD UNet mask — ship raster directly
        from typoon_inpaint.domain import EraseRaster
        import numpy as _np
        rasters = tuple(
            EraseRaster(
                x=int(em.x), y=int(em.y),
                w=int(em.image.shape[1]), h=int(em.image.shape[0]),
                data=em.image.astype(_np.uint8).tobytes(),
            )
            for em in (g.erase_masks or ())
        )

    else:
        # lens_obb / lens_aabb — ship erase_masks as compressed rasters.
        # _erase_masks_from_words already filled the tight OBB/AABB
        # via cv2.fillPoly; we preserve pixel data (not AABB rectangle).
        # zlib compress: binary rasters are ~99% compressible.
        import zlib
        from typoon_inpaint.domain import EraseRaster
        import numpy as _np
        erase_rasters = [
            EraseRaster(
                x=int(em.x), y=int(em.y),
                w=int(em.image.shape[1]), h=int(em.image.shape[0]),
                data=zlib.compress(em.image.astype(_np.uint8).tobytes(), level=1),
            )
            for em in (g.erase_masks or ())
        ]
        if erase_rasters:
            rasters = tuple(erase_rasters)
            origin  = "ctd_unet"
        else:
            # No erase masks at all despite being lens origin — treat as fallback
            origin = "polygon_fallback"

    return GroupMask(
        idx=idx, bbox=bbox,
        origin=origin, class_=class_,
        shape_kind=g.shape_kind,
        polygons=polygons, rasters=rasters,
    )


def _detect_page_kind(img: np.ndarray) -> PageKind:
    H, W = img.shape[:2]
    if H / W > 2.5:
        return "webtoon"
    import cv2
    lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.int16) - 128
    chroma = float(np.mean(np.abs(lab)))
    return "bw" if chroma < 6.0 else "color"


# ── Msgpack encoder ───────────────────────────────────────────────────────

def _encode_plan(plan: InpaintPlan) -> bytes:
    def _enc_group(g: GroupMask) -> dict:
        d: dict = {
            "idx":        g.idx,
            "bbox":       list(g.bbox),
            "origin":     g.origin,
            "class":      g.class_,
            "shape_kind": g.shape_kind,
            "polygons":   [list(p) for p in g.polygons],
            "rasters":    [
                {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "data": r.data}
                for r in g.rasters
            ],
        }
        return d

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
            "n_polygons": len(g.polygons), "n_rasters": len(g.rasters),
        }
        for g in plan.groups
    ], indent=2)
