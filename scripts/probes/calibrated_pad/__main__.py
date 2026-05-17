"""Probe driver. Re-runs detection + grouping, then re-walks the anchor
assignment to label each group as SOURCE (calibration sample) vs
FALLBACK (would receive a calibrated pad).

Imports `_spatial_join` internals — probe-only, never imported by the
production grouper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typoon.models import ModelHub  # noqa: E402
from typoon.vision._backends.comic_detr import load_session  # noqa: E402
from typoon.vision.contracts import TextBlock  # noqa: E402
from typoon.vision.detectors.lens import LensBlocksDetector  # noqa: E402
from typoon.vision.groupers._spatial_join import (  # noqa: E402
    _Anchor, _CLUSTER_CLASSES, _area, _contains_center, _dedup_anchors,
    _median_glyph_size, _word_union,
)


# Same filter range as future production code; calibration ignores
# samples outside this band.
_RATIO_PHYSICAL_MIN = 0.05
_RATIO_PHYSICAL_MAX = 0.45
_TRIM_LO = 0.10
_TRIM_HI = 0.90


@dataclass(frozen=True)
class _BubbleReport:
    idx:               int
    text:              str
    shape_kind:        str               # dialogue | burst
    role:              str               # source | fallback
    anchor_cls:        str | None        # bubble | text_bubble | text_free | None
    text_bubble_bbox:  tuple[int, int, int, int] | None
    word_union:        tuple[int, int, int, int]
    glyph_short:       int
    current_polygon:   list[list[int]]
    current_short_px:  int
    current_pad_px:    int
    proposed_short_px: int               # 0 when role=source or no ratio
    proposed_polygon:  list[list[int]]   # equal to current when no proposal
    proposed_pad_px:   int               # 0 when no proposal
    ratio_sample:      float             # glyph/text_bubble_short, source only
    delta_short_px:    int               # proposed_short - current_short


def _shape_kind_for_anchor(cls: str | None) -> str:
    if cls in ("bubble", "text_bubble"):
        return "dialogue"
    return ""  # decided per member


def _short(bbox: tuple[int, int, int, int]) -> int:
    return max(1, min(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def _polygon_short(polygon: list[list[float]]) -> int:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return max(1, int(min(max(xs) - min(xs), max(ys) - min(ys))))


def _trimmed_mean(samples: list[float], lo: float, hi: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    n = len(s)
    if n < 5:
        return sum(s) / n
    a = int(n * lo)
    b = max(a + 1, int(n * hi))
    return sum(s[a:b]) / max(1, b - a)


def _scale_bbox(bbox, scale: float):
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    hw = (bbox[2] - bbox[0]) / 2.0 * scale
    hh = (bbox[3] - bbox[1]) / 2.0 * scale
    return (int(round(cx - hw)), int(round(cy - hh)),
            int(round(cx + hw)), int(round(cy + hh)))


async def _detect_and_group(image: np.ndarray, models: Path):
    """Run production detect + group; return (blocks, anchors, groups)."""
    from typoon.vision.groupers.lens_native import LensNativeGrouper
    hub = ModelHub(models)
    detector = LensBlocksDetector(comic_detr=load_session(hub.resolve_comic_detr()))
    detection = await detector.detect(image, lang=None)

    # Re-derive anchors so we can map group → anchor.
    raw_anchors = [
        _Anchor(cls=cls, bbox=bbox, conf=conf)
        for (cls, bbox, conf) in detection.bubble_regions
        if cls in _CLUSTER_CLASSES
    ]
    text_bubbles = [a for a in raw_anchors if a.cls == "text_bubble"]
    anchors = _dedup_anchors(raw_anchors, all_text_bubbles=text_bubbles)
    anchors.sort(key=lambda a: _area(a.bbox))

    grouper = LensNativeGrouper()
    groups = await grouper.group(image, detection, None)
    return detection.blocks, anchors, groups


def _map_groups_to_anchors(
    blocks: tuple[TextBlock, ...],
    anchors: list,
    groups: tuple,
) -> dict[int, tuple[Any, list[TextBlock]]]:
    """Walk the same assignment spatial_join does, return
    {group_idx: (anchor_or_None, member_blocks)}.
    """
    # Replicate spatial_join walk: innermost-first, claim by center-in-bbox.
    assigned: set[int] = set()
    out: dict[int, tuple[Any, list[TextBlock]]] = {}
    group_aabb_by_anchor: list[tuple[tuple[int, int, int, int], Any, list[TextBlock]]] = []

    for anchor in anchors:
        member_ids = [
            i for i, b in enumerate(blocks)
            if i not in assigned and _contains_center(anchor.bbox, b.bbox)
        ]
        if not member_ids:
            continue
        members = [blocks[i] for i in member_ids]
        # The group's AABB matches `_polygon_bbox(polygon, ...)` — we
        # match groups to anchors by checking which group bbox covers
        # the anchor centre. Build the index once.
        group_aabb_by_anchor.append(
            (anchor.bbox, anchor, members)
        )
        assigned.update(member_ids)

    # Singletons
    singleton_blocks: list[TextBlock] = []
    for i, b in enumerate(blocks):
        if i not in assigned:
            singleton_blocks.append(b)

    # Now match groups by spatial proximity. spatial_join sorts the
    # final output by (bbox.y, bbox.x); we replay the same ordering.
    pre_sorted: list[tuple[tuple[int, int, int, int], Any, list[TextBlock]]] = []
    pre_sorted.extend(group_aabb_by_anchor)
    for b in singleton_blocks:
        pre_sorted.append((b.bbox, None, [b]))
    pre_sorted.sort(key=lambda it: (it[0][1], it[0][0]))

    if len(pre_sorted) != len(groups):
        # Anchors that produced no claimable blocks would drop; ignore
        # — we walk by group order and assume same count.
        pass

    for gi, g in enumerate(groups):
        # Match by closest centre. Robust to small bbox drift from
        # container expand.
        gx = (g.bbox[0] + g.bbox[2]) / 2
        gy = (g.bbox[1] + g.bbox[3]) / 2
        best = None
        best_d = 1e18
        for j, (bx, anc, mem) in enumerate(pre_sorted):
            cx = (bx[0] + bx[2]) / 2
            cy = (bx[1] + bx[3]) / 2
            d = (cx - gx) ** 2 + (cy - gy) ** 2
            if d < best_d:
                best_d = d
                best = (j, anc, mem)
        if best is not None:
            out[gi] = (best[1], best[2])
        else:
            out[gi] = (None, [])
    return out


def _build_report(groups, mapping) -> tuple[list[_BubbleReport], float, float]:
    """First pass: collect ratio samples from SOURCE groups.
    Second pass: emit per-bubble report (proposed pad for FALLBACK)."""
    body_pool: list[float] = []
    sfx_pool:  list[float] = []
    interim: list[dict] = []

    for gi, g in enumerate(groups):
        anchor, members = mapping[gi]
        glyph = _median_glyph_size(members) if members else 0
        wu = _word_union(members) if members else g.bbox

        text_bubble_bbox: tuple[int, int, int, int] | None = None
        role = "fallback"
        anchor_cls = anchor.cls if anchor is not None else None

        if anchor is not None:
            if anchor.cls == "text_bubble":
                text_bubble_bbox = tuple(anchor.bbox)
                role = "source"
            elif anchor.cls == "bubble" and anchor.inner_bbox is not None:
                text_bubble_bbox = tuple(anchor.inner_bbox)
                role = "source"

        if role == "source" and glyph > 0 and text_bubble_bbox is not None:
            tb_short = _short(text_bubble_bbox)
            if tb_short > 0:
                ratio = glyph / tb_short
                if _RATIO_PHYSICAL_MIN <= ratio <= _RATIO_PHYSICAL_MAX:
                    if g.shape_kind == "burst":
                        sfx_pool.append(ratio)
                    else:
                        body_pool.append(ratio)
        interim.append({
            "anchor_cls": anchor_cls,
            "text_bubble_bbox": text_bubble_bbox,
            "role": role,
            "wu": wu,
            "glyph": glyph,
        })

    body_r = _trimmed_mean(body_pool, _TRIM_LO, _TRIM_HI)
    sfx_r  = _trimmed_mean(sfx_pool,  _TRIM_LO, _TRIM_HI)

    reports: list[_BubbleReport] = []
    for gi, g in enumerate(groups):
        inter = interim[gi]
        cur_poly = [[int(p[0]), int(p[1])] for p in g.polygon]
        cur_short = _polygon_short(cur_poly)
        cur_pad = max(0, (cur_short - _short(inter["wu"])) // 2)

        proposed_short = 0
        proposed_poly = cur_poly
        proposed_pad = 0
        if inter["role"] == "fallback" and inter["glyph"] > 0:
            ratio = sfx_r if g.shape_kind == "burst" else body_r
            if ratio > 0:
                target_short = inter["glyph"] / ratio
                if target_short > _short(inter["wu"]):
                    # Expand word_union to target_short on the short axis only;
                    # long axis follows aspect (option B from spec). Show both
                    # short and long for now.
                    wu = inter["wu"]
                    wu_w = wu[2] - wu[0]
                    wu_h = wu[3] - wu[1]
                    if wu_w <= wu_h:
                        new_w = int(round(target_short))
                        new_h = wu_h
                    else:
                        new_w = wu_w
                        new_h = int(round(target_short))
                    cx = (wu[0] + wu[2]) / 2
                    cy = (wu[1] + wu[3]) / 2
                    new_bbox = (
                        int(round(cx - new_w / 2)), int(round(cy - new_h / 2)),
                        int(round(cx + new_w / 2)), int(round(cy + new_h / 2)),
                    )
                    proposed_poly = [
                        [new_bbox[0], new_bbox[1]],
                        [new_bbox[2], new_bbox[1]],
                        [new_bbox[2], new_bbox[3]],
                        [new_bbox[0], new_bbox[3]],
                    ]
                    proposed_short = int(round(target_short))
                    proposed_pad = max(0, (proposed_short - _short(inter["wu"])) // 2)

        ratio_sample = 0.0
        if inter["role"] == "source" and inter["text_bubble_bbox"] and inter["glyph"] > 0:
            tb_short = _short(inter["text_bubble_bbox"])
            if tb_short > 0:
                ratio_sample = round(inter["glyph"] / tb_short, 4)

        reports.append(_BubbleReport(
            idx=gi,
            text=(g.text or "")[:40],
            shape_kind=g.shape_kind,
            role=inter["role"],
            anchor_cls=inter["anchor_cls"],
            text_bubble_bbox=inter["text_bubble_bbox"],
            word_union=tuple(int(v) for v in inter["wu"]),
            glyph_short=inter["glyph"],
            current_polygon=cur_poly,
            current_short_px=cur_short,
            current_pad_px=cur_pad,
            proposed_short_px=proposed_short,
            proposed_polygon=proposed_poly,
            proposed_pad_px=proposed_pad,
            ratio_sample=ratio_sample,
            delta_short_px=proposed_short - cur_short if proposed_short else 0,
        ))
    return reports, body_r, sfx_r


def _render_overlay(image: np.ndarray, reports: list[_BubbleReport]) -> np.ndarray:
    """Yellow = current; cyan = proposed (fallback only); magenta =
    DETR text_bubble (source only). Drawn on a copy of the source.
    """
    out = image.copy()
    for r in reports:
        # Current polygon (yellow)
        pts = np.array(r.current_polygon, dtype=np.int32)
        cv2.polylines(out, [pts], True, (255, 220, 0), 2)

        # Proposed polygon (cyan) — only when fallback got a proposal
        if r.role == "fallback" and r.proposed_short_px > 0:
            pts2 = np.array(r.proposed_polygon, dtype=np.int32)
            cv2.polylines(out, [pts2], True, (0, 220, 220), 2)

        # DETR text_bubble outline (magenta) — calibration source
        if r.role == "source" and r.text_bubble_bbox is not None:
            tb = r.text_bubble_bbox
            cv2.rectangle(out, (tb[0], tb[1]), (tb[2], tb[3]), (255, 0, 220), 1)

        # Word union (thin grey)
        wu = r.word_union
        cv2.rectangle(out, (wu[0], wu[1]), (wu[2], wu[3]), (140, 140, 140), 1)

        # Label
        x, y = r.current_polygon[0]
        tag = f"#{r.idx} {r.role[0]} g={r.glyph_short}"
        if r.role == "source":
            tag += f" r={r.ratio_sample}"
        elif r.proposed_short_px:
            tag += f" Δ={r.delta_short_px:+d}"
        cv2.putText(out, tag, (max(0, x), max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 0), 1, cv2.LINE_AA)
    return out


async def main(args):
    img = np.asarray(Image.open(args.image).convert("RGB"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks, anchors, groups = await _detect_and_group(img, args.models)
    mapping = _map_groups_to_anchors(blocks, anchors, groups)
    reports, body_r, sfx_r = _build_report(groups, mapping)

    summary = {
        "image":       str(args.image),
        "page_size":   [img.shape[1], img.shape[0]],
        "n_groups":    len(groups),
        "n_source":    sum(1 for r in reports if r.role == "source"),
        "n_fallback":  sum(1 for r in reports if r.role == "fallback"),
        "n_proposed":  sum(1 for r in reports if r.proposed_short_px > 0),
        "body_ratio":  round(body_r, 4),
        "sfx_ratio":   round(sfx_r, 4),
        "bubbles":     [asdict(r) for r in reports],
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overlay = _render_overlay(img, reports)
    cv2.imwrite(str(out_dir / "overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"groups: {summary['n_groups']}  "
          f"source: {summary['n_source']}  fallback: {summary['n_fallback']}  "
          f"proposed: {summary['n_proposed']}")
    print(f"body_ratio: {summary['body_ratio']}  sfx_ratio: {summary['sfx_ratio']}")
    print(f"→ {out_dir}/calibration.json  +  overlay.png")


def cli():
    ap = argparse.ArgumentParser(prog="calibrated_pad_probe")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "debug-runs" / "calibrated_pad")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
