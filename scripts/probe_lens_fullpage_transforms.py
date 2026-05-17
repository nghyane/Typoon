"""Probe — full-page Lens with image transforms.

Test whether image-level preprocessing helps Lens detect the
`他们俩` column it currently drops. All variants run on the full
probe3 page (no crops, no upscale).

We check: do any of these recover the missing column?

  A. baseline (no preprocess)
  B. rotate 90 cw  — turns tategaki into horizontal at page level
  C. rotate 90 ccw
  D. binarize (Otsu)
  E. whiten dots (small connected components → white)
  F. sharpen (unsharp mask)
  G. upscale ×1.5
  H. invert colours

For each variant: count paragraphs Lens returns, look for `他们`
substring in any of them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402


def _binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def _whiten_dots(img: np.ndarray, max_area: int = 400) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = img.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            out[labels == i] = [255, 255, 255]
    return out


def _sharpen(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img).filter(
        ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3)
    )
    return np.asarray(pil)


def _upscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    return np.asarray(
        Image.fromarray(img).resize(
            (int(w * factor), int(h * factor)), Image.LANCZOS
        )
    )


def _invert(img: np.ndarray) -> np.ndarray:
    return 255 - img


async def _try(det, image: np.ndarray, label: str) -> None:
    res = await det.detect(image, lang="zh-Hans")
    blocks = res.blocks
    found = [b for b in blocks if "他们" in (b.text or "") or "他们俩" in (b.text or "")]
    # also any block containing 他 in zone left of x=500 (rough check)
    near = [
        b for b in blocks
        if "他" in (b.text or "") and b.bbox[0] < 500
    ]
    msg = "MATCH" if found else ("near-match" if near else "miss")
    print(
        f"[{label:30s}] blocks={len(blocks):3d}  detected_lang={res.detected_lang or 'N/A':<8s}  -> {msg}"
    )
    for b in found + near:
        print(f"   -> bbox={b.bbox}  text={b.text!r}")


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe3" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    print(f"page: {img.shape}")

    det = LensBlocksDetector()

    variants = [
        ("A. baseline", img),
        ("B. rotate 90 cw", np.rot90(img, k=-1).copy()),
        ("C. rotate 90 ccw", np.rot90(img, k=1).copy()),
        ("D. binarize (Otsu)", _binarize(img)),
        ("E. whiten dots (<400px)", _whiten_dots(img, 400)),
        ("F. whiten dots (<800px)", _whiten_dots(img, 800)),
        ("G. sharpen", _sharpen(img)),
        ("H. upscale 1.5x", _upscale(img, 1.5)),
        ("I. invert colors", _invert(img)),
        ("J. binarize + whiten", _whiten_dots(_binarize(img), 400)),
        ("K. sharpen + binarize", _binarize(_sharpen(img))),
    ]
    for label, variant_img in variants:
        await _try(det, variant_img, label)


if __name__ == "__main__":
    asyncio.run(main())
