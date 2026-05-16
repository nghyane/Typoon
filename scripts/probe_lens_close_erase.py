"""Probe — morphological closing on erase_mask.

Apply ``cv2.morphologyEx(MORPH_CLOSE)`` to the erase mask AFTER the
final isotropic dilate, before handing it off to AOT-GAN. Measure:

  * how many pixels closing adds per block (delta = closed - original)
  * how many erase-mask connected components dropped from N to 1
  * visualise side-by-side on the user's CN page

The text mask is NOT touched — render still gets per-glyph
components for font fitting.

Output:
    debug-runs/lens_bubble_probe2/probe_closing_erase.png
    (per-block: source crop | erase orig (red) | erase closed (green) | diff)
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

from typoon.vision.contracts import TextMask  # noqa: E402
from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402
from typoon.vision.groupers.lens_native import (  # noqa: E402
    LensNativeGrouper,
    _erase_dilate_px,
    _PROFILES,
    _classify_block,
)


def _close_mask(mask_img: np.ndarray, radius: int) -> np.ndarray:
    """Closing with an elliptical kernel of given radius."""
    if radius <= 0:
        return mask_img
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Pad to avoid edge bias
    pad = radius
    padded = np.pad(mask_img, pad, mode="constant", constant_values=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    return closed[pad:-pad, pad:-pad]


def _components(mask_img: np.ndarray) -> int:
    n, _, _, _ = cv2.connectedComponentsWithStats(mask_img, 8)
    return n - 1  # exclude background


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]

    det = LensBlocksDetector()
    detection = await det.detect(img, lang="zh-Hans")
    gr = LensNativeGrouper()
    groups = await gr.group(img, detection, "zh-Hans")
    print(f"groups: {len(groups)}")

    panels: list[tuple[str, np.ndarray]] = []

    for i, g in enumerate(groups):
        em = g.erase_masks[0]
        # Replicate the radius used for the original dilate
        block_class = _classify_block(  # type: ignore[call-arg]
            type("B", (), {"bbox": g.bbox, "rotation_deg": 0.0})(),
            g.text,
        )
        profile = _PROFILES[block_class]
        ts = g.typesetting
        font_px = ts.font_size_px if ts else 0
        erase_r = _erase_dilate_px(g.bbox, profile, font_px)

        # Closing kernel radius = 0.5 × erase_dilate_radius
        close_r = max(1, erase_r // 2)
        closed = _close_mask(em.image, close_r)

        delta = int(closed.sum() // 255 - em.image.sum() // 255)
        cc_before = _components(em.image)
        cc_after  = _components(closed)
        print(
            f"  [{i}] class={block_class:9s} erase_r={erase_r:2d} "
            f"close_r={close_r:2d}  cc {cc_before}->{cc_after}  +{delta}px  "
            f"text={g.text[:30]!r}"
        )

        # Build panel: crop source + 3 overlays
        x1, y1, x2, y2 = g.bbox
        PAD = 12
        cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
        cx2, cy2 = min(W, x2 + PAD), min(H, y2 + PAD)
        crop = img[cy1:cy2, cx1:cx2].copy()
        ch, cw = crop.shape[:2]

        def make_overlay(mask_img: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
            full = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
            mh, mw = mask_img.shape[:2]
            mx, my = em.x, em.y
            ix1, iy1 = max(0, mx - cx1), max(0, my - cy1)
            ix2, iy2 = min(cw, mx + mw - cx1), min(ch, my + mh - cy1)
            sx1, sy1 = ix1 - (mx - cx1), iy1 - (my - cy1)
            sx2, sy2 = sx1 + (ix2 - ix1), sy1 + (iy2 - iy1)
            full[iy1:iy2, ix1:ix2] = mask_img[sy1:sy2, sx1:sx2]
            overlay = crop.copy()
            overlay[full > 0] = color
            return cv2.addWeighted(crop, 0.45, overlay, 0.55, 0)

        # Diff: pixels added by closing
        diff = (closed.astype(np.int16) - em.image.astype(np.int16)).clip(0).astype(np.uint8)

        ov_orig   = make_overlay(em.image, (255, 60, 60))
        ov_closed = make_overlay(closed,    (60, 200, 60))
        ov_diff   = make_overlay(diff,      (255, 200, 0))

        row = np.hstack([crop, ov_orig, ov_closed, ov_diff])
        panels.append(
            (f"#{i} class={block_class} cc {cc_before}->{cc_after} +{delta}px",
             row)
        )

    # Stack panels vertically with labels
    max_w = max(p[1].shape[1] for p in panels)
    canvas_h = sum(p[1].shape[0] + 35 for p in panels) + 40
    canvas = np.full((canvas_h, max_w + 20, 3), 255, dtype=np.uint8)
    y = 20
    for label, row in panels:
        h, w = row.shape[:2]
        canvas[y:y + h, 10:10 + w] = row
        cv2.putText(
            canvas, label, (10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
        y += h + 35
    # Header
    cv2.putText(
        canvas,
        "columns: source | erase orig (red) | erase closed (green) | diff (yellow)",
        (10, canvas_h - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    out = ROOT / "debug-runs" / "lens_bubble_probe" / "probe_closing_erase.png"
    Image.fromarray(canvas).save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
