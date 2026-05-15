"""Latency benchmark: Lens vs Bing vs Offline.

Measures wall-clock at three levels:
  - single-page  : 1 page, sequential (raw detector + grouper latency)
  - concurrent-2 : 2 pages parallel (saturate session keep-alive)
  - concurrent-N : full chapter (real-world throughput)

Each preset is warmed up with one untimed call before the loop so model
load + endpoint patching doesn't pollute the first sample.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from poc_v4_e2e import _DirReader  # noqa: E402

from typoon.adapters.vision_runtime import VisionRuntimeAdapter  # noqa: E402
from typoon.stages.scan import scan_chapter  # noqa: E402
from typoon.vision.pipeline import PRESETS, VisionPipelineSpec  # noqa: E402
from typoon.vision.runtime import build_vision_runtime  # noqa: E402


CHAPTER = ROOT / "tests" / "fixtures" / "sample_chapters" / "ch001"


# ─── Helpers ──────────────────────────────────────────────────────────────


async def _scan(spec: VisionPipelineSpec, reader: _DirReader) -> tuple[float, int]:
    """Run one full chapter scan; return (elapsed_s, n_bubbles)."""
    adapter, _, _ = VisionRuntimeAdapter.from_config(source_lang="en")
    runtime = build_vision_runtime(
        spec, models_dir=adapter.hub.dir, source_lang="en",
    )
    t0 = time.perf_counter()
    result = await scan_chapter(
        reader.chapter(), reader, runtime, source_lang="en",
    )
    elapsed = time.perf_counter() - t0
    return elapsed, len(result.bubble_records())


async def _detector_only(
    spec: VisionPipelineSpec, image: np.ndarray, n_calls: int = 3,
) -> list[float]:
    """Measure detector-only latency (skip grouper/recognizer/eraser)."""
    adapter, _, _ = VisionRuntimeAdapter.from_config(source_lang="en")
    runtime = build_vision_runtime(
        spec, models_dir=adapter.hub.dir, source_lang="en",
    )
    # Warmup
    await runtime.detector.detect(image, "en")
    samples: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        await runtime.detector.detect(image, "en")
        samples.append(time.perf_counter() - t0)
    return samples


def _summary(samples: list[float]) -> str:
    if not samples:
        return "no samples"
    p50 = statistics.median(samples) * 1000
    p_max = max(samples) * 1000
    p_min = min(samples) * 1000
    mean = statistics.mean(samples) * 1000
    return f"min={p_min:5.0f}ms  p50={p50:5.0f}ms  mean={mean:5.0f}ms  max={p_max:5.0f}ms"


# ─── Bench cases ──────────────────────────────────────────────────────────


async def bench_detector_latency(spec_name: str, image: np.ndarray) -> None:
    spec = VisionPipelineSpec.preset(spec_name)
    samples = await _detector_only(spec, image, n_calls=3)
    print(f"  {spec_name:8}  {_summary(samples)}")


async def bench_chapter(spec_name: str, reader: _DirReader, page_concurrency: int) -> None:
    base = VisionPipelineSpec.preset(spec_name)
    spec = base.with_overrides(page_concurrency=page_concurrency)
    elapsed, n_bubbles = await _scan(spec, reader)
    n_pages = reader.page_count
    per_page = elapsed / max(1, n_pages)
    print(
        f"  {spec_name:8}  pc={page_concurrency} "
        f"total={elapsed:5.2f}s  per_page={per_page:5.2f}s  bubbles={n_bubbles}"
    )


# ─── Driver ───────────────────────────────────────────────────────────────


async def main() -> None:
    if not CHAPTER.is_dir():
        print(f"missing fixture: {CHAPTER}")
        return

    paths = sorted(CHAPTER.glob("*.webp"))
    if not paths:
        print(f"no webp in {CHAPTER}")
        return

    reader = _DirReader(paths)

    # Load a representative page once
    import cv2
    image = cv2.imread(str(paths[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nchapter: {CHAPTER.name}  pages: {len(paths)}")

    presets_to_bench = ["lens", "bing"]

    print(f"\n=== single-page detector latency (3 samples, warmed) ===")
    for name in presets_to_bench:
        try:
            await bench_detector_latency(name, image)
        except Exception as e:
            print(f"  {name:8}  FAIL: {e}")

    for pc in (1, 2, 4):
        print(f"\n=== full chapter, page_concurrency={pc} ===")
        for name in presets_to_bench:
            try:
                await bench_chapter(name, reader, pc)
            except Exception as e:
                print(f"  {name:8}  FAIL: {e}")


if __name__ == "__main__":
    asyncio.run(main())
