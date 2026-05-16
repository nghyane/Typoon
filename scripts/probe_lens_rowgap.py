"""Probe A — row-gap aware glyph mask.

Hypothesis: when Lens partially recognizes a row (returns a short line
covering only part of the visual row), build the row's mask span across
the full block width instead of only the word bboxes.

Detection signal: ``line_width / median(other line widths) < THRESHOLD``,
with the suspicious line being neither first nor last in the block
(edges are often legitimately short — punctuation, justification).

For each suspicious line, paint a full-block-width band at the line's
y-range into the glyph base, then run normal dilate + in-bounds clip.

Compares 3 mask variants side-by-side on the problem bubble from
debug-runs/lens_bubble_probe2 (the "难道······他 才是" block):

    A) current     — word bboxes only
    B) row-aware   — stretch to block width for short non-edge lines
    C) reference   — entire block rect (overkill but ground truth)

Output: debug-runs/lens_bubble_probe2/probe_A_rowgap.png — RGB grid
showing source crop, A overlay, B overlay, C overlay.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.contracts import TextBlock  # noqa: E402
from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402
from typoon.vision.groupers.lens_native import (  # noqa: E402
    _EXPAND_PAD_MIN_PX,
    _EXPAND_PAD_X_FRACTION,
    _EXPAND_PAD_Y_FRACTION,
    _build_typesetting_hint,
)


# ─── Tuning constants for probe A ────────────────────────────────────────


_ROW_SHORT_RATIO = 0.5    # line_width < ratio × median(other widths) → suspicious
_ROW_MIN_LINES   = 3      # need at least this many lines to apply (signal noise)


# ─── Mask builders ───────────────────────────────────────────────────────


def build_mask_current(block: TextBlock) -> np.ndarray:
    """Variant A — current production: word bboxes only."""
    x1, y1, x2, y2 = block.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    if not block.words:
        return np.full((h, w), 255, dtype=np.uint8)

    ts = _build_typesetting_hint(block)
    font_px = ts.font_size_px if ts else 0
    pad_x = max(_EXPAND_PAD_MIN_PX, int(font_px * _EXPAND_PAD_X_FRACTION))
    pad_y = max(_EXPAND_PAD_MIN_PX, int(font_px * _EXPAND_PAD_Y_FRACTION))

    base = np.zeros((h, w), dtype=np.uint8)
    in_bounds = np.zeros((h, w), dtype=np.uint8)
    for word in block.words:
        wx1, wy1, wx2, wy2 = word.bbox
        lx1 = max(0, min(w, wx1 - x1))
        ly1 = max(0, min(h, wy1 - y1))
        lx2 = max(0, min(w, wx2 - x1))
        ly2 = max(0, min(h, wy2 - y1))
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        base[ly1:ly2, lx1:lx2] = 255
        ex1 = max(0, min(w, wx1 - x1 - pad_x))
        ey1 = max(0, min(h, wy1 - y1 - pad_y))
        ex2 = max(0, min(w, wx2 - x1 + pad_x))
        ey2 = max(0, min(h, wy2 - y1 + pad_y))
        in_bounds[ey1:ey2, ex1:ex2] = 255

    dilate_radius = max(1, int(font_px * 0.10))
    ksize = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(base, kernel, iterations=1)
    return np.where(in_bounds > 0, dilated, 0).astype(np.uint8)


def _suspicious_line_indices(block: TextBlock) -> list[int]:
    """Lines whose width is anomalously short vs the rest of the block.

    Excludes first/last lines — those are often legitimately short
    (closing punctuation, ragged justification). The median is computed
    over the OTHER lines so a single short outlier doesn't suppress
    the signal.
    """
    lines = block.lines
    if len(lines) < _ROW_MIN_LINES:
        return []
    widths = [max(1, l.bbox[2] - l.bbox[0]) for l in lines]
    out: list[int] = []
    for i in range(1, len(lines) - 1):
        others = widths[:i] + widths[i + 1:]
        med = statistics.median(others)
        if widths[i] / med < _ROW_SHORT_RATIO:
            out.append(i)
    return out


def build_mask_rowgap(block: TextBlock) -> tuple[np.ndarray, list[int]]:
    """Variant B — row-aware: stretch short non-edge lines to block width."""
    x1, y1, x2, y2 = block.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    if not block.words:
        return np.full((h, w), 255, dtype=np.uint8), []

    ts = _build_typesetting_hint(block)
    font_px = ts.font_size_px if ts else 0
    pad_x = max(_EXPAND_PAD_MIN_PX, int(font_px * _EXPAND_PAD_X_FRACTION))
    pad_y = max(_EXPAND_PAD_MIN_PX, int(font_px * _EXPAND_PAD_Y_FRACTION))

    base = np.zeros((h, w), dtype=np.uint8)
    in_bounds = np.zeros((h, w), dtype=np.uint8)
    for word in block.words:
        wx1, wy1, wx2, wy2 = word.bbox
        lx1 = max(0, min(w, wx1 - x1))
        ly1 = max(0, min(h, wy1 - y1))
        lx2 = max(0, min(w, wx2 - x1))
        ly2 = max(0, min(h, wy2 - y1))
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        base[ly1:ly2, lx1:lx2] = 255
        ex1 = max(0, min(w, wx1 - x1 - pad_x))
        ey1 = max(0, min(h, wy1 - y1 - pad_y))
        ex2 = max(0, min(w, wx2 - x1 + pad_x))
        ey2 = max(0, min(h, wy2 - y1 + pad_y))
        in_bounds[ey1:ey2, ex1:ex2] = 255

    suspicious = _suspicious_line_indices(block)
    for idx in suspicious:
        ly1 = max(0, block.lines[idx].bbox[1] - y1)
        ly2 = min(h, block.lines[idx].bbox[3] - y1)
        if ly2 <= ly1:
            continue
        # Stretch base AND in_bounds across full block width for this row
        base[ly1:ly2, :] = 255
        # Bleed in_bounds by pad_y so dilate at the row edges isn't clipped
        in_bounds[
            max(0, ly1 - pad_y):min(h, ly2 + pad_y), :
        ] = 255

    dilate_radius = max(1, int(font_px * 0.10))
    ksize = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(base, kernel, iterations=1)
    refined = np.where(in_bounds > 0, dilated, 0).astype(np.uint8)
    return refined, suspicious


def build_mask_block_rect(block: TextBlock) -> np.ndarray:
    """Variant C — reference upper bound: entire block rect."""
    x1, y1, x2, y2 = block.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return np.full((h, w), 255, dtype=np.uint8)


# ─── Visualisation ───────────────────────────────────────────────────────


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha=0.55) -> np.ndarray:
    out = rgb.copy()
    overlay = out.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)


def compose_grid(crops: list[tuple[str, np.ndarray]]) -> np.ndarray:
    # Each crop assumed same size
    h, w = crops[0][1].shape[:2]
    pad = 40
    grid = np.full(
        ((h + pad) * len(crops) + pad, w + pad * 2, 3),
        255, dtype=np.uint8,
    )
    for i, (label, im) in enumerate(crops):
        y = pad + (h + pad) * i
        grid[y:y + h, pad:pad + w] = im
        cv2.putText(
            grid, label, (pad, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return grid


# ─── Driver ──────────────────────────────────────────────────────────────


async def main() -> None:
    src_path = ROOT / "debug-runs" / "lens_bubble_probe2" / "source.png"
    if not src_path.exists():
        print(f"source not found: {src_path}")
        return

    img = np.asarray(Image.open(src_path).convert("RGB"))
    H, W = img.shape[:2]
    print(f"page {W}x{H}")

    det = LensBlocksDetector()
    detection = await det.detect(img, lang=None)
    target = next(
        (b for b in detection.blocks if "难道他才是" in (b.text or "")),
        None,
    )
    if target is None:
        print("target block not found")
        return

    x1, y1, x2, y2 = target.bbox
    print(f"target bbox {target.bbox} ({x2-x1}x{y2-y1})")
    print(f"lines={len(target.lines)} words={len(target.words)}")
    for i, l in enumerate(target.lines):
        print(f"  L[{i}] w={l.bbox[2]-l.bbox[0]} text={l.text!r}")

    mask_a = build_mask_current(target)
    mask_b, suspicious = build_mask_rowgap(target)
    mask_c = build_mask_block_rect(target)
    print(f"suspicious lines: {suspicious}")
    print(f"mask coverage A={mask_a.sum()/255:.0f}px  "
          f"B={mask_b.sum()/255:.0f}px  C={mask_c.sum()/255:.0f}px")

    # Crop area for visualisation (a bit larger than block)
    PAD = 30
    cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
    cx2, cy2 = min(W, x2 + PAD), min(H, y2 + PAD)
    crop = img[cy1:cy2, cx1:cx2]

    def place(mask, color):
        h, w = mask.shape[:2]
        full = np.zeros((cy2 - cy1, cx2 - cx1, 3), dtype=np.uint8)
        rgb = crop.copy()
        # paint mask on rgb
        m_full = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
        m_full[y1 - cy1:y1 - cy1 + h, x1 - cx1:x1 - cx1 + w] = mask
        return overlay_mask(rgb, m_full, color)

    grid = compose_grid([
        ("source", crop),
        ("A: word-bbox (current)", place(mask_a, (255, 0, 0))),
        ("B: row-gap aware", place(mask_b, (0, 200, 0))),
        ("C: full block rect (overkill ref)", place(mask_c, (0, 0, 255))),
    ])
    out_path = src_path.parent / "probe_A_rowgap.png"
    Image.fromarray(grid).save(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
