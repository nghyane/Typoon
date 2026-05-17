"""Probe — detector-guided Lens (per-region OCR vs tile sliding).

Strategy: replace 4 sliding tiles with per-region Lens calls. Each
region comes from comic_detr's text_bubble + text_free detections.

We compare:
  A. Current: 4 sliding tiles of (page_w × 900) → Lens
  B. Per-region: ~12 crops of detected text regions → Lens

For each: count recovered text, missed text, latency budget.
On probe3 verifies that `他们俩` is recovered without any preprocessing.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typoon.vision.detectors.lens_blocks import LensBlocksDetector  # noqa: E402

CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}
CROP_PAD_PX = 8


def _load_comic_detr():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "ogkalu/comic-text-and-bubble-detector", "detector-v4-s_int8.onnx",
    )
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _comic_detect(sess, img: np.ndarray, conf: float = 0.3) -> list[dict]:
    h, w = img.shape[:2]
    resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(
        None,
        {"images": arr, "orig_target_sizes": np.array([[w, h]], dtype=np.int64)},
    )
    labels, boxes, scores = out
    above = scores[0] > conf
    dets = []
    for i in np.where(above)[0]:
        x1, y1, x2, y2 = boxes[0][i].tolist()
        dets.append({
            "class": int(labels[0][i]),
            "conf": float(scores[0][i]),
            "bbox": (
                max(0, int(x1) - CROP_PAD_PX),
                max(0, int(y1) - CROP_PAD_PX),
                min(w, int(x2) + CROP_PAD_PX),
                min(h, int(y2) + CROP_PAD_PX),
            ),
        })
    return dets


async def _lens_crop(api, crop: np.ndarray, lang: str = "zh-Hans") -> list[str]:
    """OCR a single crop with Lens. Returns paragraph texts."""
    result = await api.process_image(
        crop, ocr_language=lang, output_format="detailed",
    )
    paras = result.get("detailed_blocks") or []
    texts = []
    for p in paras:
        t = (p.get("text") or "").replace("\n", " ").strip()
        if t:
            texts.append(t)
    return texts


async def main() -> None:
    src = ROOT / "debug-runs" / "lens_bubble_probe3" / "source.png"
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]
    print(f"page: {W}x{H}")

    # ─── A. Baseline: current full-page Lens ──────────────────────────
    print("\n=== A. Baseline (current sliding-tile Lens) ===")
    det = LensBlocksDetector()
    t0 = time.perf_counter()
    res = await det.detect(img, lang="zh-Hans")
    elapsed_a = time.perf_counter() - t0
    print(f"  {len(res.blocks)} blocks, {elapsed_a*1000:.0f}ms")
    texts_a = {(b.text or "").strip() for b in res.blocks if b.text}
    has_target = any("他们" in t for t in texts_a)
    print(f"  has 他们: {has_target}")

    # ─── B. Detector-guided per-region Lens ────────────────────────────
    print("\n=== B. comic_detr per-region Lens ===")
    sess = _load_comic_detr()
    t_det = time.perf_counter()
    comic_dets = _comic_detect(sess, img)
    det_ms = (time.perf_counter() - t_det) * 1000
    text_regions = [d for d in comic_dets if d["class"] in (1, 2)]
    print(f"  comic_detr: {det_ms:.0f}ms, "
          f"{len(text_regions)} text regions "
          f"(text_bubble={sum(1 for d in text_regions if d['class']==1)}, "
          f"text_free={sum(1 for d in text_regions if d['class']==2)})")

    api = await det._get_api()  # noqa: SLF001
    t_ocr = time.perf_counter()
    # Run all crops in parallel
    crops = [img[d["bbox"][1]:d["bbox"][3], d["bbox"][0]:d["bbox"][2]] for d in text_regions]
    results = await asyncio.gather(*[_lens_crop(api, c) for c in crops])
    ocr_ms = (time.perf_counter() - t_ocr) * 1000
    print(f"  Lens (parallel): {ocr_ms:.0f}ms over {len(crops)} regions")

    texts_b: set[str] = set()
    for region, region_texts in zip(text_regions, results):
        for t in region_texts:
            texts_b.add(t)
    print(f"  total unique texts: {len(texts_b)}")
    has_target_b = any("他们" in t for t in texts_b)
    print(f"  has 他们: {has_target_b}")

    elapsed_b = (det_ms + ocr_ms) / 1000
    print(f"\n  Total B: {elapsed_b*1000:.0f}ms")

    # ─── Diff ──────────────────────────────────────────────────────────
    only_a = texts_a - texts_b
    only_b = texts_b - texts_a
    print(f"\n=== DIFF ===")
    print(f"Only in A ({len(only_a)}): {sorted(only_a)[:5]}")
    print(f"Only in B ({len(only_b)}): {sorted(only_b)[:5]}")
    print(f"Latency: A={elapsed_a*1000:.0f}ms  B={elapsed_b*1000:.0f}ms"
          f"  ({'B' if elapsed_b < elapsed_a else 'A'} wins)")


if __name__ == "__main__":
    asyncio.run(main())
