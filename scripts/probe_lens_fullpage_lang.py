"""Probe — Lens full-page detect: auto vs language hint.

For the CN probe2 page (5 horizontal CN dialogue blocks), compare:
    A) ocr_language=""        (current: auto)
    B) ocr_language="zh-Hans" (proposed: explicit hint when source is known)

For the CN probe1 page (tategaki), same comparison.

Goal: verify hint at minimum matches auto, ideally recovers blocks
auto missed (especially the `难道······他 才是` row in probe2).

Output: print kept/rejected counts and any text diffs per page.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import (  # noqa: E402
    LensBlocksDetector,
    _ocr_tile,
)


async def _detect_with_lang(
    detector: LensBlocksDetector, img: np.ndarray, lang_hint: str,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Bypass detector.detect to inject a custom ocr_language per tile."""
    # We use the public detect() path but monkey-patch _ocr_tile via lang
    # threading would be cleaner — for the probe, just re-implement.
    from typoon.vision.detectors.lens_blocks import _iter_tiles, _dedup_raw

    api = await detector._get_api()  # noqa: SLF001
    h, w = img.shape[:2]

    tiles = list(_iter_tiles(img))
    # Manually pass ocr_language to each tile
    results = []
    for origin_y, tile in tiles:
        try:
            result = await api.process_image(
                tile,
                ocr_language=lang_hint,
                output_format="detailed",
            )
        except Exception as e:
            print(f"  tile y={origin_y} failed: {e}")
            continue
        from typoon.vision.detectors.lens_blocks import _paragraph_to_raw
        tile_h = tile.shape[0]
        for paragraph in result.get("detailed_blocks") or []:
            block = _paragraph_to_raw(paragraph, origin_y, w, tile_h)
            if block is not None:
                results.append(block)
    deduped = _dedup_raw(results)
    return [(b.text, b.bbox) for b in deduped]


async def compare_page(label: str, path: Path, lang_hint: str) -> None:
    img = np.asarray(Image.open(path).convert("RGB"))
    print(f"\n=== {label}  hint={lang_hint!r} ===")
    det = LensBlocksDetector()
    blocks = await _detect_with_lang(det, img, lang_hint)
    print(f"  {len(blocks)} blocks")
    for t, b in blocks:
        x1, y1, x2, y2 = b
        print(f"    [{x1:4d},{y1:4d},{x2:4d},{y2:4d}] {t!r}")


async def main() -> None:
    probe2 = ROOT / "debug-runs" / "lens_bubble_probe2" / "source.png"
    probe1 = ROOT / "debug-runs" / "lens_bubble_probe" / "source.png"
    # Probe2: CN horizontal
    await compare_page("probe2 CN horizontal", probe2, "")
    await compare_page("probe2 CN horizontal", probe2, "zh-Hans")
    # Probe1: CN tategaki
    await compare_page("probe1 CN vertical",   probe1, "")
    await compare_page("probe1 CN vertical",   probe1, "zh-Hans")


if __name__ == "__main__":
    asyncio.run(main())
