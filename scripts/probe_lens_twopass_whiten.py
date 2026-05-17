"""Probe — two-pass detection: baseline + whiten-dots fallback.

Strategy:
  Pass 1: standard Lens detect on the page.
  Pass 2: same page with small dark connected components whitened
          (suppresses dense punctuation runs that confuse Lens
          tategaki classifier).
  Merge:  keep all Pass 1 blocks. From Pass 2, only add blocks whose
          bbox does not overlap any Pass 1 block above an IoU
          threshold — these are the recovery candidates.

Verify on probe3 (target: recover `他们俩` column).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402


def _whiten_dots(img: np.ndarray, max_area: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = img.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            out[labels == i] = [255, 255, 255]
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(aa, bb)   # intersection-over-smaller


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe3" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    print(f"page: {img.shape}")

    det = LensBlocksDetector()

    # Pass 1: baseline
    pass1 = await det.detect(img, lang="zh-Hans")
    print(f"\nPass 1 (baseline): {len(pass1.blocks)} blocks")

    # Pass 2: whiten small dots
    img2 = _whiten_dots(img, max_area=400)
    pass2 = await det.detect(img2, lang="zh-Hans")
    print(f"Pass 2 (whiten <400px): {len(pass2.blocks)} blocks")

    # Merge: keep all Pass 1, add Pass 2 only if not overlapping
    OVERLAP_THRESHOLD = 0.3
    pass1_bboxes = [b.bbox for b in pass1.blocks]
    new_recoveries = []
    for b2 in pass2.blocks:
        max_overlap = max(
            (_iou(b2.bbox, b1) for b1 in pass1_bboxes),
            default=0.0,
        )
        if max_overlap < OVERLAP_THRESHOLD:
            new_recoveries.append((max_overlap, b2))

    print(f"\nRecovered blocks from Pass 2 (overlap < {OVERLAP_THRESHOLD}):")
    for ov, b in new_recoveries:
        print(f"  bbox={b.bbox}  overlap={ov:.2f}  text={b.text!r}")

    # Sanity: blocks in Pass 1 but missing in Pass 2 (would be lost)
    pass2_bboxes = [b.bbox for b in pass2.blocks]
    lost = []
    for b1 in pass1.blocks:
        max_overlap = max(
            (_iou(b1.bbox, b2) for b2 in pass2_bboxes),
            default=0.0,
        )
        if max_overlap < OVERLAP_THRESHOLD:
            lost.append((max_overlap, b1))
    print(f"\nBlocks in Pass 1 not in Pass 2 (would be lost if we replaced Pass 1):")
    for ov, b in lost:
        print(f"  bbox={b.bbox}  text={b.text!r}")


if __name__ == "__main__":
    asyncio.run(main())
