"""Probe — why does Lens drop the 他们俩 column?

The bubble around `他们俩` in probe3 has a tategaki column composed of:
    top:    `…` × N stacked dots
    middle: 他 / 们 / 俩 (3 CJK glyphs)
    bottom: `…` × N stacked dots

Lens consistently drops this column in every test so far (full-page,
full-bubble crop, every ocr_language). Hypothesis: the column is
>50% punctuation/decoration and Lens's classifier discards it as
noise. We probe by varying the crop to isolate the trigger:

  1. Crop just the 3 CJK glyphs (no surrounding dots).
  2. Crop with progressively fewer dots above + below.
  3. Crop with dots replaced by empty space.

Output: per-variant Lens text + paragraph count + writing_direction.
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


# Bubble area in probe3 source.png — eyeballed from masks.png
BUBBLE_X1, BUBBLE_X2 = 180, 530
BUBBLE_Y1, BUBBLE_Y2 = 80, 720

# The 他们俩 column lives roughly in this x-range inside the bubble
COL_X1, COL_X2 = 365, 445


async def _ocr(api, crop: np.ndarray, label: str, lang: str = "") -> None:
    h, w = crop.shape[:2]
    # Upscale to >= 200px short side (match production row recovery path)
    if min(h, w) < 200:
        scale = max(1, int(np.ceil(200 / min(h, w))))
        pil = Image.fromarray(crop).resize((w * scale, h * scale), Image.LANCZOS)
        crop = np.asarray(pil)
    result = await api.process_image(crop, ocr_language=lang, output_format="detailed")
    raw = result.get("raw_response_objects")
    n = len(raw.text.text_layout.paragraphs) if raw else 0
    WD = {0: "LTR", 1: "RTL", 2: "T2B"}
    print(f"\n[{label}]  shape={crop.shape}  paragraphs={n}")
    for p in (raw.text.text_layout.paragraphs if raw else []):
        text = "|".join("".join(w.plain_text for w in l.words) for l in p.lines)
        wd = WD.get(p.writing_direction, "?")
        print(f"  wd={wd}  text={text!r}")


async def main() -> None:
    img = np.asarray(
        Image.open(ROOT / "debug-runs" / "lens_bubble_probe3" / "source.png").convert("RGB")
    )
    H, W = img.shape[:2]
    print(f"page {W}x{H}")

    det = LensBlocksDetector()
    api = await det._get_api()  # noqa: SLF001

    # Full bubble baseline
    bubble = img[BUBBLE_Y1:BUBBLE_Y2, BUBBLE_X1:BUBBLE_X2].copy()
    await _ocr(api, bubble, "0. full bubble")

    # Column only (with all dots)
    col = img[BUBBLE_Y1:BUBBLE_Y2, COL_X1:COL_X2].copy()
    await _ocr(api, col, "1. column only (full, all dots)")
    await _ocr(api, col, "1b. column only (lang=zh-Hans)", lang="zh-Hans")

    # Just the 3 CJK glyphs region (eyeballed: y ~280..500)
    glyphs_only = img[280:500, COL_X1:COL_X2].copy()
    await _ocr(api, glyphs_only, "2. glyphs only (no dots)")

    # Glyphs + few dots above
    short_top = img[200:500, COL_X1:COL_X2].copy()
    await _ocr(api, short_top, "3. glyphs + few dots top")

    # Glyphs + few dots both sides
    short_both = img[200:550, COL_X1:COL_X2].copy()
    await _ocr(api, short_both, "4. glyphs + few dots both sides")

    # Replace dots with white space — manufacture a "clean" version
    clean = img[BUBBLE_Y1:BUBBLE_Y2, COL_X1:COL_X2].copy()
    # Whiten top dots (y range in crop coords: 0..200, since BUBBLE_Y1=80)
    clean[:200, :] = 255
    clean[420:, :] = 255  # whiten bottom dots
    await _ocr(api, clean, "5. dots whited out")


if __name__ == "__main__":
    asyncio.run(main())
