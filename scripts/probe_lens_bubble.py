"""Probe: run lens_blocks detector + lens_native grouper on one image
and dump debug artifacts (per-stage visual + JSON).

Usage:
    python scripts/probe_lens_bubble.py <image_path> [--out debug-runs/lens_bubble_probe]

Outputs under <out>/:
    source.png          — input as RGB png
    detect.json         — kept blocks + rejected (with reason)
    detect.png          — blocks drawn over input with text + class hint
    group.json          — final BubbleGroups
    group.png           — groups drawn over input
    masks.png           — text + erase mask union overlays
    merge_edges.json    — pairwise tategaki merge candidates and decisions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.contracts import BubbleGroup, DetectionResult, TextBlock  # noqa: E402
from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402
from typoon.vision.groupers.lens_native import (  # noqa: E402
    LensNativeGrouper,
    _classify_block,
    _compatible_with_cluster,
    _x_gap,
    _y_overlap_ratio,
)


def _load_image(p: Path) -> np.ndarray:
    img = Image.open(p).convert("RGB")
    return np.asarray(img)


def _draw_blocks(
    canvas: np.ndarray,
    blocks: list[TextBlock],
    rejected: list[tuple[TextBlock, str]],
) -> np.ndarray:
    out = canvas.copy()
    # Rejected: thin red
    for b, reason in rejected:
        x1, y1, x2, y2 = b.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            out, reason, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA,
        )
    # Kept: green, with class
    for b in blocks:
        klass = _classify_block(b, b.text or "")
        color = {
            "sfx": (255, 140, 0),
            "dialogue": (0, 180, 0),
            "narration": (0, 120, 255),
        }[klass]
        x1, y1, x2, y2 = b.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{klass} rot={b.rotation_deg:.1f} L={len(b.lines)} W={len(b.words)}"
        cv2.putText(
            out, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return out


def _draw_groups(canvas: np.ndarray, groups: tuple[BubbleGroup, ...]) -> np.ndarray:
    out = canvas.copy()
    for i, g in enumerate(groups):
        color = (255, 0, 255) if g.text_direction == "vertical" else (0, 200, 255)
        x1, y1, x2, y2 = g.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        ts = g.typesetting
        ts_s = (
            f"fs={ts.font_size_px} L={ts.line_count} a={ts.avg_chars_per_line:.1f}"
            if ts else "ts=None"
        )
        label = f"#{i} {g.shape_kind} {g.text_direction} {ts_s}"
        cv2.putText(
            out, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return out


def _paint_mask_overlay(
    canvas: np.ndarray, groups: tuple[BubbleGroup, ...]
) -> np.ndarray:
    out = canvas.copy()
    h, w = canvas.shape[:2]
    erase_canvas = np.zeros((h, w), dtype=np.uint8)
    text_canvas = np.zeros((h, w), dtype=np.uint8)
    for g in groups:
        for tm in g.text_masks:
            _stamp_mask(text_canvas, tm)
        for em in g.erase_masks:
            _stamp_mask(erase_canvas, em)
    # red = erase (broad), green = text (tight)
    overlay = out.copy()
    overlay[erase_canvas > 0] = (255, 0, 0)
    overlay[text_canvas > 0] = (0, 255, 0)
    return cv2.addWeighted(out, 0.55, overlay, 0.45, 0)


def _stamp_mask(target: np.ndarray, mask) -> None:
    H, W = target.shape[:2]
    mx, my = mask.x, mask.y
    mh, mw = mask.image.shape[:2]
    x1, y1 = max(0, mx), max(0, my)
    x2, y2 = min(W, mx + mw), min(H, my + mh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - mx, y1 - my
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    sub = mask.image[sy1:sy2, sx1:sx2]
    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], sub)


def _block_to_dict(b: TextBlock) -> dict:
    return {
        "bbox": list(b.bbox),
        "confidence": round(b.confidence, 4),
        "text": b.text or "",
        "rotation_deg": round(b.rotation_deg, 2),
        "n_lines": len(b.lines),
        "n_words": len(b.words),
        "lines": [
            {"bbox": list(l.bbox), "text": l.text,
             "rotation_deg": round(l.rotation_deg, 2)}
            for l in b.lines
        ],
        "class": _classify_block(b, b.text or ""),
    }


def _group_to_dict(g: BubbleGroup) -> dict:
    return {
        "bbox": list(g.bbox),
        "text": g.text,
        "confidence": round(g.confidence, 4),
        "source": g.source,
        "shape_kind": g.shape_kind,
        "used_fallback": g.used_fallback,
        "rotation_deg": round(g.rotation_deg, 2),
        "text_direction": g.text_direction,
        "typesetting": (
            None if g.typesetting is None
            else {
                "font_size_px": g.typesetting.font_size_px,
                "line_count": g.typesetting.line_count,
                "avg_chars_per_line": round(g.typesetting.avg_chars_per_line, 2),
            }
        ),
        "n_text_masks": len(g.text_masks),
        "n_erase_masks": len(g.erase_masks),
    }


def _merge_candidates(blocks: list[TextBlock]) -> list[dict]:
    """Replicate cluster-compat predicate over candidate vertical pairs."""
    from typoon.vision.groupers.lens_native import (
        _build_typesetting_hint,
        _infer_text_direction,
    )
    # Build minimal BubbleGroup-like shim for compatibility checker
    class _Shim:
        __slots__ = ("bbox",)
        def __init__(self, b): self.bbox = b
    diagnostics: list[dict] = []
    info = []
    for i, b in enumerate(blocks):
        ts = _build_typesetting_hint(b)
        d = _infer_text_direction(b, ts)
        info.append((i, b, d))
    for i, bi, di in info:
        for j, bj, dj in info:
            if j <= i:
                continue
            if di != "vertical" or dj != "vertical":
                continue
            overlap_ratio = _y_overlap_ratio(bi.bbox, bj.bbox)
            gap = _x_gap(bi.bbox, bj.bbox)
            compat = _compatible_with_cluster(_Shim(bi.bbox), [_Shim(bj.bbox)])
            diagnostics.append({
                "i": i, "j": j,
                "text_i": bi.text, "text_j": bj.text,
                "bbox_i": list(bi.bbox), "bbox_j": list(bj.bbox),
                "y_overlap_ratio": round(overlap_ratio, 3),
                "x_gap_px": int(gap),
                "merged": bool(compat),
            })
    return diagnostics


async def _run(image_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = _load_image(image_path)
    h, w = img_rgb.shape[:2]
    Image.fromarray(img_rgb).save(out_dir / "source.png")
    print(f"loaded {image_path.name}: {w}x{h}")

    detector = LensBlocksDetector()
    print("running Lens detector…")
    detection: DetectionResult = await detector.detect(img_rgb, lang=None)
    print(
        f"  kept={len(detection.blocks)} rejected={len(detection.rejected)} "
        f"detected_lang={detection.detected_lang}"
    )

    detect_blob = {
        "page_size": list(detection.page_size),
        "detected_lang": detection.detected_lang,
        "n_kept": len(detection.blocks),
        "n_rejected": len(detection.rejected),
        "kept": [_block_to_dict(b) for b in detection.blocks],
        "rejected": [
            {**_block_to_dict(b), "reason": r}
            for (b, r) in detection.rejected
        ],
    }
    (out_dir / "detect.json").write_text(
        json.dumps(detect_blob, indent=2, ensure_ascii=False)
    )

    detect_png = _draw_blocks(img_rgb, list(detection.blocks), list(detection.rejected))
    Image.fromarray(detect_png).save(out_dir / "detect.png")

    # Merge diagnostics (pre-group)
    merges = _merge_candidates(list(detection.blocks))
    (out_dir / "merge_edges.json").write_text(
        json.dumps(merges, indent=2, ensure_ascii=False)
    )
    merged_n = sum(1 for m in merges if m["merged"])
    print(f"  vertical-pair candidates: {len(merges)}, merge=True: {merged_n}")

    print("running lens_native grouper…")
    grouper = LensNativeGrouper()
    groups = await grouper.group(img_rgb, detection, lang=None)
    print(f"  groups={len(groups)}")

    group_blob = {
        "n_groups": len(groups),
        "groups": [_group_to_dict(g) for g in groups],
    }
    (out_dir / "group.json").write_text(
        json.dumps(group_blob, indent=2, ensure_ascii=False)
    )
    Image.fromarray(_draw_groups(img_rgb, groups)).save(out_dir / "group.png")
    Image.fromarray(_paint_mask_overlay(img_rgb, groups)).save(out_dir / "masks.png")

    # Summary
    by_class: dict[str, int] = {}
    for b in detection.blocks:
        c = _classify_block(b, b.text or "")
        by_class[c] = by_class.get(c, 0) + 1
    by_dir: dict[str, int] = {}
    for g in groups:
        by_dir[g.text_direction] = by_dir.get(g.text_direction, 0) + 1
    summary = {
        "image": str(image_path),
        "page_size": [w, h],
        "detector": {
            "kept": len(detection.blocks),
            "rejected": len(detection.rejected),
            "by_class": by_class,
            "rejected_reasons": _counter(detection.rejected),
        },
        "grouper": {
            "groups": len(groups),
            "by_text_direction": by_dir,
            "merged_pairs": merged_n,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"artifacts → {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _counter(rejected):
    out: dict[str, int] = {}
    for _, r in rejected:
        out[r] = out.get(r, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "debug-runs" / "lens_bubble_probe",
    )
    args = ap.parse_args()
    asyncio.run(_run(args.image, args.out))


if __name__ == "__main__":
    main()
