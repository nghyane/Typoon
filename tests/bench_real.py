"""Lightweight real scan benchmark — scanner (det + platform OCR), no LaMa.

Usage: .venv/bin/python tests/bench_real.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "training" / "manga" / "jujutsu-kaisen"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CHAPTER = "ch024.3"
MAX_PAGES = 3


class DirSource:
    def __init__(self, path: Path, max_pages: int = 0) -> None:
        self._path = path
        self._files: list[Path] = []
        self._max = max_pages

    async def fetch(self) -> None:
        self._files = sorted(self._path.glob("*.webp"))
        if self._max:
            self._files = self._files[:self._max]

    def page_count(self) -> int:
        return len(self._files)

    def load_page(self, index: int) -> np.ndarray:
        img = cv2.imread(str(self._files[index]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


async def main():
    needed = ["ppocr-det.safetensors", "ppocr-det-config.json"]
    missing = [n for n in needed if not (MODELS_DIR / n).exists()]
    if missing:
        print(f"⚠ Missing: {missing}")
        return

    if not (DATA_ROOT / CHAPTER).exists():
        print(f"⚠ Not found: {DATA_ROOT / CHAPTER}")
        return

    from typoon.vision.scanner import create_scanner
    from typoon.models import ModelHub

    print("Loading scan models (det 14MB)...")
    t0 = time.monotonic()
    hub = ModelHub(MODELS_DIR)
    scanner = create_scanner(hub=hub)
    print(f"Loaded in {(time.monotonic() - t0)*1000:.0f}ms\n")

    source = DirSource(DATA_ROOT / CHAPTER, max_pages=MAX_PAGES)
    await source.fetch()
    n = source.page_count()
    print(f"{CHAPTER}: scanning {n} pages")
    print("-" * 50)

    page_times: list[float] = []
    total_bubbles = 0

    for i in range(n):
        img = source.load_page(i)
        h, w = img.shape[:2]

        t0 = time.monotonic()
        scanned = scanner.scan(img)
        scan_ms = (time.monotonic() - t0) * 1000

        bubbles = len(scanned)
        page_times.append(scan_ms)
        total_bubbles += bubbles
        print(f"  page {i}: {w}x{h} | {scan_ms:.0f}ms | {bubbles} bubbles")
        for sb in scanned[:3]:
            print(f"    [{sb.confidence:.0%}] {sb.text[:60]}")
        del img

    avg = sum(page_times) / len(page_times)
    print(f"\nAvg/page: {avg:.0f}ms | Total: {sum(page_times):.0f}ms | Bubbles: {total_bubbles}")

    full_chapter_pages = 10
    scan_est = avg * full_chapter_pages
    print(f"\n--- Overlap estimate ({full_chapter_pages}-page chapter) ---")
    print(f"Scan/chapter ≈ {scan_est/1000:.1f}s")
    for tr_s in [2, 5, 10]:
        tr = tr_s * 1000
        overlap = min(scan_est, tr)
        pct = overlap / (scan_est + tr) * 100
        print(f"  Translate ~{tr_s}s → save {overlap/1000:.1f}s ({pct:.0f}% of serial)")


if __name__ == "__main__":
    asyncio.run(main())
