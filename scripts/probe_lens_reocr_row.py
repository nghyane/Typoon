"""Probe — re-OCR suspicious row via Lens.

Hypothesis: when ``_suspicious_line_indices`` flags a row in a Lens
block (e.g. line width ≪ median because Lens missed glyphs around an
ellipsis), re-running Lens on a tight crop of that row recovers the
missing text. The crop is small (1 line tall, full block width) so
Lens has no excuse to mis-segment.

Compares on the problem block from debug-runs/lens_bubble_probe2:
    target row L[2] before: "才是"
    target row L[2] after  : (should contain "难道...他 才是")

Output: prints original vs re-OCRed text for the suspicious row, and
saves a 2-row visual:
    debug-runs/lens_bubble_probe2/probe_reocr_row.png
"""

from __future__ import annotations

import asyncio
import io
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.contracts import TextBlock  # noqa: E402
from typoon.vision.detectors.lens_blocks import (  # noqa: E402
    LensBlocksDetector,
)


_ROW_SHORT_RATIO = 0.5
_ROW_MIN_LINES   = 3
# Vertical breathing room around the row crop — Lens needs a bit of
# padding above/below the glyphs or it under-segments.
_CROP_PAD_Y_FRAC = 0.45        # × line height
_CROP_PAD_X_PX   = 6


def _suspicious_line_indices(block: TextBlock) -> list[int]:
    if len(block.lines) < _ROW_MIN_LINES:
        return []
    widths = [max(1, l.bbox[2] - l.bbox[0]) for l in block.lines]
    out: list[int] = []
    for i in range(1, len(block.lines) - 1):
        others = widths[:i] + widths[i + 1:]
        med = statistics.median(others)
        if widths[i] / med < _ROW_SHORT_RATIO:
            out.append(i)
    return out


def _row_crop_bbox(
    block: TextBlock, line_idx: int, page_w: int, page_h: int,
) -> tuple[int, int, int, int]:
    """Build a crop that spans the full block width, padded vertically."""
    line = block.lines[line_idx]
    lh = max(1, line.bbox[3] - line.bbox[1])
    pad_y = int(lh * _CROP_PAD_Y_FRAC)
    x1 = max(0, block.bbox[0] - _CROP_PAD_X_PX)
    x2 = min(page_w, block.bbox[2] + _CROP_PAD_X_PX)
    y1 = max(0, line.bbox[1] - pad_y)
    y2 = min(page_h, line.bbox[3] + pad_y)
    return (x1, y1, x2, y2)


async def _reocr_crop(api, crop_rgb: np.ndarray) -> str:
    """Call Lens directly on a crop, return concatenated text."""
    result = await api.process_image(
        crop_rgb,
        ocr_language="",
        output_format="detailed",
    )
    paragraphs = result.get("detailed_blocks") or []
    chunks: list[str] = []
    for p in paragraphs:
        t = (p.get("text") or "").replace("\n", " ").strip()
        if t:
            chunks.append(t)
    return "  ".join(chunks)


async def main() -> None:
    src_path = ROOT / "debug-runs" / "lens_bubble_probe2" / "source.png"
    if not src_path.exists():
        print(f"source not found: {src_path}")
        return

    img = np.asarray(Image.open(src_path).convert("RGB"))
    page_h, page_w = img.shape[:2]
    print(f"page {page_w}x{page_h}")

    det = LensBlocksDetector()
    detection = await det.detect(img, lang=None)
    target = next(
        (b for b in detection.blocks if "难道他才是" in (b.text or "")),
        None,
    )
    if target is None:
        print("target block not found")
        return

    print(f"target block bbox: {target.bbox}")
    print(f"original full text: {target.text!r}")
    print("per-line widths:")
    for i, l in enumerate(target.lines):
        w = l.bbox[2] - l.bbox[0]
        print(f"  L[{i}] w={w:4d}  text={l.text!r}")

    sus = _suspicious_line_indices(target)
    print(f"\nsuspicious lines: {sus}")
    if not sus:
        print("no suspicious lines, abort")
        return

    # Set up Lens API directly (re-use the same patched endpoint as the
    # detector — detector caches the API instance after first detect()).
    api = await det._get_api()  # noqa: SLF001  intentional reach for probe

    visuals: list[tuple[str, np.ndarray, str]] = []
    for idx in sus:
        cx1, cy1, cx2, cy2 = _row_crop_bbox(target, idx, page_w, page_h)
        crop = img[cy1:cy2, cx1:cx2].copy()
        print(f"\nrow L[{idx}] crop: ({cx1},{cy1},{cx2},{cy2}) "
              f"shape={crop.shape}")

        # Pad to Lens minimum size if too small (Lens prefers >=200px shortest)
        h, w = crop.shape[:2]
        if min(h, w) < 200:
            # Upscale ×3 — matches manga_ocr behaviour for tiny crops
            scale = max(1, int(np.ceil(200 / min(h, w))))
            crop_pil = Image.fromarray(crop)
            crop_pil = crop_pil.resize(
                (w * scale, h * scale), Image.LANCZOS,
            )
            crop = np.asarray(crop_pil)
            print(f"  upscaled ×{scale} -> {crop.shape}")

        text_new = await _reocr_crop(api, crop)
        original_line = target.lines[idx].text
        print(f"  original line text : {original_line!r}")
        print(f"  re-OCR row text    : {text_new!r}")
        visuals.append((f"L[{idx}] orig={original_line!r}",
                        crop, text_new))

    # Compose visual: stack each suspicious row crop with labels
    if visuals:
        pad = 30
        max_w = max(v[1].shape[1] for v in visuals)
        total_h = sum(v[1].shape[0] + pad * 2 for v in visuals) + pad
        canvas = np.full((total_h, max_w + pad * 2, 3), 255, dtype=np.uint8)
        import cv2
        y = pad
        for label, crop, new in visuals:
            h, w = crop.shape[:2]
            canvas[y:y + h, pad:pad + w] = crop
            cv2.putText(
                canvas, label, (pad, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"new = {new[:50]}", (pad, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA,
            )
            y += h + pad * 2
        out_path = src_path.parent / "probe_reocr_row.png"
        Image.fromarray(canvas).save(out_path)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
