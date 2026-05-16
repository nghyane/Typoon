"""Probe — re-OCR row with explicit language hint vs auto.

Tests whether passing the language detected on the first full-page
pass (``detection.detected_lang``) to the row re-OCR helps or hurts.

Variants on the problem row L[2] from probe2:
    1. ocr_language=""        (auto, current probe)
    2. ocr_language="zh-Hans" (page-level detected lang)
    3. ocr_language="zh"      (BCP-47 prefix only)

Expected: an explicit lang hint should be at worst neutral and at best
prevents the recognizer from drifting on a single-row crop where the
auto-detector has less context to anchor on.
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


async def _ocr(api, crop, lang: str, label: str) -> None:
    result = await api.process_image(
        crop, ocr_language=lang, output_format="detailed",
    )
    paras = result.get("detailed_blocks") or []
    print(f"\n[{label}] ocr_language={lang!r}  paragraphs={len(paras)}")
    for p in paras:
        for l in p.get("lines") or []:
            t = (l.get("text") or "").strip()
            if t:
                print(f"  L: {t!r}")


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe2" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]

    det = LensBlocksDetector()
    detection = await det.detect(img, lang=None)
    print(f"page-level detected_lang: {detection.detected_lang!r}")

    target = next(b for b in detection.blocks if "难道他才是" in (b.text or ""))
    x1, y1, x2, y2 = target.bbox
    line = target.lines[2]
    pad_y = int((line.bbox[3] - line.bbox[1]) * 0.45)
    crop = img[
        max(0, line.bbox[1] - pad_y):min(H, line.bbox[3] + pad_y),
        max(0, x1 - 6):min(W, x2 + 6),
    ].copy()
    h, w = crop.shape[:2]
    scale = max(1, int(np.ceil(200 / min(h, w))))
    crop_up = np.asarray(
        Image.fromarray(crop).resize((w * scale, h * scale), Image.LANCZOS)
    )
    print(f"row crop shape={crop_up.shape}")

    api = await det._get_api()  # noqa: SLF001
    for lang_arg, label in [
        ("",        "auto"),
        ("zh-Hans", "zh-Hans"),
        ("zh",      "zh"),
        ("zh-CN",   "zh-CN"),
        ("ja",      "ja (wrong, control)"),
    ]:
        await _ocr(api, crop_up, lang_arg, label)


if __name__ == "__main__":
    asyncio.run(main())
