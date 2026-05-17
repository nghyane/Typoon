"""Lens bubble pass (Phase B) — authoritative OCR per DETR region.

Walks Comic-DETR text_bubble / bubble / text_free anchors and decides
which ones the tile pass already covered fully, which are empty, and
which only have partial coverage. Empty + partial anchors are re-OCRed
on a single tight crop — that crop gives Lens the full glyph context
in one shot, fixing both tategaki direction mistakes and edge-glyph
drops that the tile pass produced.

Anchor priority per spatially-clustered group:
    text_bubble  >  bubble  >  text_free
``text_bubble`` is the tightest inner rect, so when DETR surfaces one
we use it. When DETR only emits ``bubble`` we use that. ``text_free``
catches captions outside balloons.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image as _PILImage

from ...contracts import TextBlock
from .geometry import Frame, paragraph_to_raw
from .tile_pass import _raw_paragraphs


__all__ = ["run"]

logger = logging.getLogger(__name__)


# Anchor selection.
_ANCHOR_PRECEDENCE = ("text_bubble", "bubble", "text_free")
_CLUSTER_IOU = 0.5            # single-link IoU for clustering DETR regions
_GAP_THRESHOLD_FACTOR = 0.7   # gap > 0.7 × median_line_h ⇒ partial

# Crop config.
_CROP_PAD_PX = 6
_MIN_CROP_DIM = 200           # Lens recognition collapses below ~200 px short side


class Action(Enum):
    COMPLETE = "complete"     # tile pass already covered this anchor
    EMPTY    = "empty"        # no Lens block lands inside
    PARTIAL  = "partial"      # block(s) inside but Lens missed a chunk


@dataclass(frozen=True, slots=True)
class _Anchor:
    cls:  str
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class _Diagnosis:
    anchor:    _Anchor
    action:    Action
    member_ix: list[int]      # indices into the input ``blocks`` list


# ─── Public entry ─────────────────────────────────────────────────────────


async def run(
    api,
    image: np.ndarray,
    blocks: list[TextBlock],
    regions: tuple[tuple[str, tuple[int, int, int, int], float], ...],
    lang_hint: str,
) -> list[TextBlock]:
    """Replace blocks inside incomplete anchors with crop-OCR results."""
    if not regions:
        return blocks

    H, W = image.shape[:2]
    anchors = _select_anchors(regions)
    if not anchors:
        return blocks

    diagnoses = [_diagnose(a, blocks) for a in anchors]
    todo = [d for d in diagnoses if d.action is not Action.COMPLETE]
    logger.info(
        "bubble pass: %d anchors → empty=%d partial=%d complete=%d",
        len(anchors),
        sum(1 for d in diagnoses if d.action is Action.EMPTY),
        sum(1 for d in diagnoses if d.action is Action.PARTIAL),
        sum(1 for d in diagnoses if d.action is Action.COMPLETE),
    )
    if not todo:
        return blocks

    recovered_per_anchor = await asyncio.gather(*[
        _ocr_anchor(api, image, d.anchor, (W, H), lang_hint) for d in todo
    ])
    return _splice(blocks, todo, recovered_per_anchor)


# ─── Anchor selection ────────────────────────────────────────────────────


def _select_anchors(
    regions: tuple[tuple[str, tuple[int, int, int, int], float], ...],
) -> list[_Anchor]:
    """One anchor per spatial cluster, picked by class precedence."""
    clusters = _cluster_regions(regions, _CLUSTER_IOU)
    out: list[_Anchor] = []
    for cluster in clusters:
        winner = _pick_cluster_anchor(cluster)
        if winner is not None:
            out.append(winner)
    return out


def _cluster_regions(
    regions, iou_thr: float,
) -> list[list[tuple[str, tuple[int, int, int, int], float]]]:
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, (_, ai, _) in enumerate(regions):
        for j in range(i + 1, len(regions)):
            _, aj, _ = regions[j]
            if _iou(ai, aj) > iou_thr:
                union(i, j)

    buckets: dict[int, list] = {}
    for i, r in enumerate(regions):
        buckets.setdefault(find(i), []).append(r)
    return list(buckets.values())


def _pick_cluster_anchor(cluster) -> _Anchor | None:
    for cls in _ANCHOR_PRECEDENCE:
        same = [r for r in cluster if r[0] == cls]
        if same:
            best = max(same, key=lambda r: r[2])
            return _Anchor(cls=best[0], bbox=best[1])
    return None


# ─── Diagnosis ───────────────────────────────────────────────────────────


def _diagnose(anchor: _Anchor, blocks: list[TextBlock]) -> _Diagnosis:
    member_ix = [
        i for i, b in enumerate(blocks)
        if _center_inside(b.bbox, anchor.bbox)
    ]
    if not member_ix:
        return _Diagnosis(anchor, Action.EMPTY, [])

    members = [blocks[i] for i in member_ix]
    wu = _word_union(members)
    line_h = _median_line_height(members)
    if line_h == 0:
        # No line geometry → cannot decide, leave alone.
        return _Diagnosis(anchor, Action.COMPLETE, member_ix)

    thr = _GAP_THRESHOLD_FACTOR * line_h
    gap_top    = wu[1] - anchor.bbox[1]
    gap_bot    = anchor.bbox[3] - wu[3]
    gap_left   = wu[0] - anchor.bbox[0]
    gap_right  = anchor.bbox[2] - wu[2]
    if max(gap_top, gap_bot, gap_left, gap_right) > thr:
        return _Diagnosis(anchor, Action.PARTIAL, member_ix)
    return _Diagnosis(anchor, Action.COMPLETE, member_ix)


# ─── OCR on a single anchor crop ─────────────────────────────────────────


async def _ocr_anchor(
    api,
    image: np.ndarray,
    anchor: _Anchor,
    page_size: tuple[int, int],
    lang_hint: str,
) -> list[TextBlock]:
    W, H = page_size
    bx1, by1, bx2, by2 = anchor.bbox
    x1 = max(0, bx1 - _CROP_PAD_PX)
    y1 = max(0, by1 - _CROP_PAD_PX)
    x2 = min(W, bx2 + _CROP_PAD_PX)
    y2 = min(H, by2 + _CROP_PAD_PX)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    scale = 1
    if min(h, w) < _MIN_CROP_DIM:
        scale = max(1, int(np.ceil(_MIN_CROP_DIM / min(h, w))))
        pil = _PILImage.fromarray(crop).resize(
            (w * scale, h * scale), _PILImage.LANCZOS,
        )
        crop = np.asarray(pil)

    try:
        result = await api.process_image(
            crop, ocr_language=lang_hint, output_format="detailed",
        )
    except Exception as e:
        logger.warning("bubble re-OCR failed @ %s: %s", anchor.bbox, e)
        return []

    frame = Frame(
        origin_x=x1, origin_y=y1,
        frame_w=crop.shape[1], frame_h=crop.shape[0], scale=scale,
    )
    raw_paragraphs = _raw_paragraphs(result)
    out: list[TextBlock] = []
    for i, paragraph in enumerate(result.get("detailed_blocks") or []):
        raw_para = raw_paragraphs[i] if i < len(raw_paragraphs) else None
        rb = paragraph_to_raw(paragraph, raw_para, frame, page_size)
        if rb is None:
            continue
        out.append(TextBlock(
            bbox=rb.bbox, polygon=None, confidence=rb.confidence,
            text=rb.text, detector="lens_blocks/bubble",
            rotation_deg=rb.rotation_deg,
            words=rb.words, lines=rb.lines,
            text_direction=rb.text_direction,
        ))
    logger.info(
        "  anchor %s %s → %d block(s)",
        anchor.cls, anchor.bbox, len(out),
    )
    return out


# ─── Splice recovered blocks back into the keep list ─────────────────────


def _splice(
    blocks: list[TextBlock],
    diagnoses: list[_Diagnosis],
    recovered_per: list[list[TextBlock]],
) -> list[TextBlock]:
    """Drop replaced members; append recovered blocks. EMPTY drops nothing."""
    drop: set[int] = set()
    for d, rec in zip(diagnoses, recovered_per):
        if not rec:
            continue  # Lens returned nothing — keep originals as-is
        drop.update(d.member_ix)
    additions: list[TextBlock] = []
    for d, rec in zip(diagnoses, recovered_per):
        additions.extend(rec)
    return [b for i, b in enumerate(blocks) if i not in drop] + additions


# ─── Geometry helpers ────────────────────────────────────────────────────


def _center_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int],
) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def _word_union(members: list[TextBlock]) -> tuple[int, int, int, int]:
    boxes = [w.bbox for m in members for w in m.words]
    if not boxes:
        boxes = [m.bbox for m in members]
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def _median_line_height(members: list[TextBlock]) -> int:
    heights: list[int] = []
    for m in members:
        for ln in m.lines:
            short = min(
                max(1, ln.bbox[2] - ln.bbox[0]),
                max(1, ln.bbox[3] - ln.bbox[1]),
            )
            heights.append(short)
    if not heights:
        return 0
    return int(statistics.median(heights))
