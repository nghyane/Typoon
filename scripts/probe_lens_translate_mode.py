"""Probe — alternative Lens techniques besides image preprocessing.

Questions to answer:

  1. Does Lens translate-mode (with target_translation_language) see
     glyphs that OCR-mode drops?
  2. Do different region/timezone/surface settings change recognition?
  3. Does Lens's "interaction" RPC (second endpoint) re-process the
     image with different settings?

If any of these recover the `他们俩` column WITHOUT image preprocessing,
that's a cleaner production fix.
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


async def _ocr_with_translate(api, img_bytes, w, h, *, target=None, lang=""):
    """Call Lens with optional translate target. Direct protobuf path."""
    from chrome_lens_py.core.protobuf_builder import create_ocr_translate_request
    payload, uuid_ = create_ocr_translate_request(
        image_bytes=img_bytes,
        width=w, height=h,
        ocr_language=lang,
        target_translation_language=target,
    )
    proto = await api.request_handler.send_request(payload, uuid_)
    if not (proto.HasField("objects_response")
            and proto.objects_response.HasField("text")
            and proto.objects_response.text.HasField("text_layout")):
        return [], None
    layout = proto.objects_response.text.text_layout
    paras = []
    for p in layout.paragraphs:
        text = " | ".join(
            "".join(word.plain_text + word.text_separator for word in line.words).strip()
            for line in p.lines
        )
        paras.append((text, p.geometry.bounding_box.center_x))
    return paras, proto.objects_response.text.content_language


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe3" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]
    print(f"page: {W}x{H}")

    # Encode page once
    import io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    det = LensBlocksDetector()
    api = await det._get_api()  # noqa: SLF001

    print("\n=== Direct OCR-only (baseline) ===")
    paras, lang = await _ocr_with_translate(api, img_bytes, W, H, target=None, lang="zh-Hans")
    print(f"  {len(paras)} paragraphs, lang={lang!r}")
    found = [t for t, _ in paras if "他们" in t]
    if found:
        for t in found:
            print(f"  ★ MATCH: {t!r}")

    print("\n=== OCR + translate target=vi ===")
    paras, lang = await _ocr_with_translate(api, img_bytes, W, H, target="vi", lang="zh-Hans")
    print(f"  {len(paras)} paragraphs, lang={lang!r}")
    found = [t for t, _ in paras if "他们" in t]
    if found:
        for t in found:
            print(f"  ★ MATCH: {t!r}")

    print("\n=== OCR + translate target=en ===")
    paras, lang = await _ocr_with_translate(api, img_bytes, W, H, target="en", lang="zh-Hans")
    print(f"  {len(paras)} paragraphs, lang={lang!r}")
    found = [t for t, _ in paras if "他们" in t]
    if found:
        for t in found:
            print(f"  ★ MATCH: {t!r}")

    print("\n=== OCR + translate target=ja (force tategaki path) ===")
    paras, lang = await _ocr_with_translate(api, img_bytes, W, H, target="ja", lang="zh-Hans")
    print(f"  {len(paras)} paragraphs, lang={lang!r}")
    found = [t for t, _ in paras if "他们" in t]
    if found:
        for t in found:
            print(f"  ★ MATCH: {t!r}")


if __name__ == "__main__":
    asyncio.run(main())
