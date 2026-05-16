"""Probe — re-OCR full block vs re-OCR single row.

Question: when Lens misses glyphs in a row of a block, does re-OCRing
the **whole block crop** recover them? Or only the **row crop** trick
works?

Hypothesis A (row wins): the issue is Lens needing more pixel headroom;
upscaling a single row to ≥200px gives the recognizer enough surface
area, while re-running the same-resolution block gives identical output.

Hypothesis B (block wins): the issue is just network non-determinism /
context bias; re-running on a fresh block crop with similar upscale
also recovers the text — cheaper because 1 call covers all rows.

We test on the same problem block from probe2.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402


_BLOCK_PAD_PX = 8


async def _ocr_crop(api, crop_rgb: np.ndarray, label: str) -> dict:
    result = await api.process_image(
        crop_rgb,
        ocr_language="",
        output_format="detailed",
    )
    paragraphs = result.get("detailed_blocks") or []
    lines_out: list[str] = []
    para_texts: list[str] = []
    for p in paragraphs:
        t = (p.get("text") or "").strip()
        if t:
            para_texts.append(t)
        for l in p.get("lines") or []:
            lt = (l.get("text") or "").strip()
            if lt:
                lines_out.append(lt)
    print(f"\n[{label}] paragraphs={len(paragraphs)} lines={len(lines_out)}")
    for i, t in enumerate(lines_out):
        print(f"  L[{i}] {t!r}")
    return {"paragraphs": para_texts, "lines": lines_out}


async def main() -> None:
    src_path = ROOT / "debug-runs" / "lens_bubble_probe2" / "source.png"
    img = np.asarray(Image.open(src_path).convert("RGB"))
    H, W = img.shape[:2]

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
    print(f"target block bbox {target.bbox} -> {x2-x1}x{y2-y1}")
    print(f"original full text: {target.text!r}")

    api = await det._get_api()  # noqa: SLF001

    # Variant 1 — block crop, native resolution
    bx1 = max(0, x1 - _BLOCK_PAD_PX)
    by1 = max(0, y1 - _BLOCK_PAD_PX)
    bx2 = min(W, x2 + _BLOCK_PAD_PX)
    by2 = min(H, y2 + _BLOCK_PAD_PX)
    block_native = img[by1:by2, bx1:bx2].copy()
    print(f"\nblock native crop shape={block_native.shape}")
    await _ocr_crop(api, block_native, "block native")

    # Variant 2 — block crop, upscaled to short_side=400
    h, w = block_native.shape[:2]
    target_short = 400
    scale = max(1, int(np.ceil(target_short / min(h, w))))
    pil = Image.fromarray(block_native).resize(
        (w * scale, h * scale), Image.LANCZOS
    )
    block_up = np.asarray(pil)
    print(f"\nblock upscaled ×{scale} -> {block_up.shape}")
    await _ocr_crop(api, block_up, f"block upscaled x{scale}")

    # Variant 3 — single row L[2], upscaled (already verified working)
    line = target.lines[2]
    pad_y = int((line.bbox[3] - line.bbox[1]) * 0.45)
    rx1 = max(0, x1 - 6)
    rx2 = min(W, x2 + 6)
    ry1 = max(0, line.bbox[1] - pad_y)
    ry2 = min(H, line.bbox[3] + pad_y)
    row_native = img[ry1:ry2, rx1:rx2].copy()
    h, w = row_native.shape[:2]
    scale = max(1, int(np.ceil(200 / min(h, w))))
    pil = Image.fromarray(row_native).resize(
        (w * scale, h * scale), Image.LANCZOS
    )
    row_up = np.asarray(pil)
    print(f"\nrow upscaled ×{scale} -> {row_up.shape}")
    await _ocr_crop(api, row_up, f"row L[2] upscaled x{scale}")


if __name__ == "__main__":
    asyncio.run(main())
