"""Benchmark: pipeline optimizations with simulated delays.

Compares sequential vs queue-based series pipeline, verifies progressive output
timing, and validates model lifecycle behavior.

Usage: .venv/bin/python tests/bench_pipeline.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from typoon.config import Config, ProviderConfig
from typoon.app.events import (
    ChapterDone,
    ChapterStart,
    Event,
    Hook,
    ModelsUnloaded,
    TranslateDone,
    TranslateStart,
    TranslationReady,
)
from typoon.domain.bubble import Bubble, Page, Session
from typoon.vision.chapter_images import ChapterImages


# ── Simulated delays (seconds) ───────────────────────────────────────

SCAN_DELAY = 0.15       # MLX detect+OCR per chapter
TRANSLATE_DELAY = 0.30  # LLM network round-trip per chapter
ERASE_DELAY = 0.10      # LaMa inpainting per chapter
KNOWLEDGE_DELAY = 0.05  # Knowledge consolidation per chapter
NUM_CHAPTERS = 4
BUBBLES_PER_PAGE = 3
PAGES_PER_CHAPTER = 2


# ── Event recorder ───────────────────────────────────────────────────

@dataclass
class TimedEvent:
    ts: float
    name: str
    detail: str = ""


class RecorderHook(Hook):
    def __init__(self) -> None:
        self.events: list[TimedEvent] = []
        self._t0 = time.monotonic()

    def on(self, event: Event) -> None:
        elapsed = time.monotonic() - self._t0
        name = type(event).__name__
        detail = ""
        match event:
            case ChapterStart(chapter=ch):
                detail = f"ch={ch}"
            case TranslationReady(translated=tr, total=n):
                detail = f"{tr}/{n}"
            case ChapterDone(elapsed=e):
                detail = f"{e:.2f}s"
            case ModelsUnloaded(stage=s):
                detail = s
        self.events.append(TimedEvent(ts=elapsed, name=name, detail=detail))


# ── Mock models with delays ──────────────────────────────────────────

class FakeScanner:
    """Simulates scanner.scan() → list[ScannedBubble]."""

    def scan(self, image: np.ndarray) -> list:
        time.sleep(SCAN_DELAY / PAGES_PER_CHAPTER)
        from typoon.vision.scanner import ScannedBubble
        return [
            ScannedBubble(
                polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
                text=f"text_{i}", confidence=0.95,
            )
            for i in range(BUBBLES_PER_PAGE)
        ]


class FakeEraser:
    def erase(self, canvas, masks):
        time.sleep(ERASE_DELAY / PAGES_PER_CHAPTER)
        return canvas


class FakeSource:
    def __init__(self, n_pages: int = PAGES_PER_CHAPTER) -> None:
        self._n = n_pages

    def page_count(self) -> int:
        return self._n

    def load_page(self, index: int) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    async def fetch(self) -> None:
        pass


class FakeStore:
    """Store that tracks knowledge saves for gating verification."""
    def __init__(self) -> None:
        self.knowledge_log: list[tuple[int, float]] = []
        self._knowledge: dict[int, str] = {}

    async def get_project(self, sid):
        return {"source_lang": "en", "target_lang": "vi"}

    async def get_glossary(self, sid):
        return {}

    async def get_knowledge(self, sid, before_chapter):
        for ch in sorted(self._knowledge.keys(), reverse=True):
            if ch < before_chapter:
                return self._knowledge[ch]
        return None

    async def save_knowledge(self, sid, chapter, snapshot):
        self._knowledge[chapter] = snapshot
        self.knowledge_log.append((chapter, time.monotonic()))

    async def save_translations(self, sid, ch, bubbles):
        pass

    async def get_chapter_translations(self, sid, ch):
        return []

    async def glossary_search(self, sid, q):
        return []

    async def glossary_upsert(self, sid, s, t, n):
        pass

    async def add_note(self, sid, ch, nt, c):
        pass

    async def search_context(self, sid, q, scope, limit):
        return []

    async def get_chapter_pairs(self, sid, ch):
        return []

    async def add_chapter(self, sid, ch, **kw):
        pass

    async def set_chapter_status(self, sid, ch, status):
        pass

    async def get_chapter_status(self, sid, ch):
        return None

    async def get_chapter_retry_count(self, sid, ch):
        return 0

    async def increment_retry_count(self, sid, ch):
        pass

    async def delete_chapter_data(self, sid, ch):
        pass


# ── Patched Engine that uses fake models ─────────────────────────────

def _make_engine():
    from typoon.engine import Engine
    engine = Engine.__new__(Engine)
    engine.scanner = FakeScanner()
    engine.eraser = FakeEraser()
    engine._hub = None
    return engine


def _make_pages() -> list[Page]:
    """Create pre-built pages with bubbles (skip real scan)."""
    pages = []
    for pi in range(PAGES_PER_CHAPTER):
        bubbles = [
            Bubble(idx=bi, page_index=pi, source_text=f"text_{bi}",
                   polygon=[[0, 0], [100, 0], [100, 50], [0, 50]])
            for bi in range(BUBBLES_PER_PAGE)
        ]
        pages.append(Page(index=pi, bubbles=bubbles))
    return pages


# ── Benchmark 1: Sequential series (baseline) ───────────────────────

async def bench_sequential(n_chapters: int) -> tuple[float, RecorderHook]:
    """Simulate old sequential: scan → translate → erase → knowledge, per chapter."""
    hook = RecorderHook()
    t0 = time.monotonic()

    for ch in range(1, n_chapters + 1):
        hook.on(ChapterStart(chapter=ch, pages=PAGES_PER_CHAPTER))

        # Scan
        time.sleep(SCAN_DELAY)

        # Translate (simulate async LLM)
        total = PAGES_PER_CHAPTER * BUBBLES_PER_PAGE
        hook.on(TranslateStart(total_bubbles=total))
        await asyncio.sleep(TRANSLATE_DELAY)
        hook.on(TranslateDone(translated=total, total=total, turns=1))
        hook.on(TranslationReady(pages=PAGES_PER_CHAPTER, translated=total, total=total))

        # Erase
        time.sleep(ERASE_DELAY)

        # Knowledge
        await asyncio.sleep(KNOWLEDGE_DELAY)
        hook.on(ChapterDone(elapsed=time.monotonic() - t0))

    elapsed = time.monotonic() - t0
    return elapsed, hook


# ── Benchmark 2: Queue-based pipeline (optimized) ───────────────────

async def bench_queue_pipeline(n_chapters: int) -> tuple[float, RecorderHook]:
    """Simulate new queue pipeline: scan(N+1) overlaps with translate(N)."""
    hook = RecorderHook()
    t0 = time.monotonic()

    chapters = list(range(1, n_chapters + 1))

    async def simulate_scan(ch: int) -> tuple[int, float]:
        hook.on(ChapterStart(chapter=ch, pages=PAGES_PER_CHAPTER))
        await asyncio.sleep(0)
        time.sleep(SCAN_DELAY)
        return ch, time.monotonic()

    async def simulate_translate(ch: int) -> None:
        total = PAGES_PER_CHAPTER * BUBBLES_PER_PAGE
        hook.on(TranslateStart(total_bubbles=total))
        await asyncio.sleep(TRANSLATE_DELAY)
        hook.on(TranslateDone(translated=total, total=total, turns=1))
        hook.on(TranslationReady(pages=PAGES_PER_CHAPTER, translated=total, total=total))

    # Scan first chapter
    await simulate_scan(chapters[0])

    for i, ch in enumerate(chapters):
        next_scan = None
        if i + 1 < len(chapters):
            next_scan = asyncio.create_task(simulate_scan(chapters[i + 1]))

        await simulate_translate(ch)

        if next_scan is not None:
            await next_scan

        time.sleep(ERASE_DELAY)
        await asyncio.sleep(KNOWLEDGE_DELAY)
        hook.on(ChapterDone(elapsed=time.monotonic() - t0))

    elapsed = time.monotonic() - t0
    return elapsed, hook


# ── Benchmark 3: Model lifecycle ─────────────────────────────────────

async def bench_model_lifecycle() -> dict:
    """Verify ensure/unload behavior on Engine."""
    from typoon.engine import Engine

    engine = Engine.__new__(Engine)
    engine.scanner = FakeScanner()
    engine.eraser = FakeEraser()
    engine._hub = None

    results = {}

    # ensure_* is no-op when models are loaded
    engine.ensure_scan_models()
    engine.ensure_erase_models()
    results["ensure_noop"] = engine.scanner is not None and engine.eraser is not None

    # unload sets to None
    hook = RecorderHook()
    engine.unload_scan_models(hook)
    results["scan_unloaded"] = engine.scanner is None
    results["unload_event"] = any(e.name == "ModelsUnloaded" and e.detail == "scan" for e in hook.events)

    engine.unload_erase_models(hook)
    results["erase_unloaded"] = engine.eraser is None

    # ensure_* without hub raises
    try:
        engine.ensure_scan_models()
        results["ensure_no_hub_raises"] = False
    except RuntimeError:
        results["ensure_no_hub_raises"] = True

    return results


# ── Benchmark 4: Progressive output timing ──────────────────────────

async def bench_progressive_output() -> dict:
    """Verify TranslationReady fires between translate and erase."""
    from unittest.mock import patch

    from typoon.app.workflows.project import ResumePolicy, run_pipeline

    hook = RecorderHook()
    store = FakeStore()
    engine = _make_engine()
    config = Config(providers={"openai": ProviderConfig(type="openai", endpoint="http://fake")})

    async def mock_translate_pages(pages, session):
        await asyncio.sleep(0.05)
        for p in pages:
            for b in p.bubbles:
                b.translated_text = f"translated_{b.idx}"
        return 1, None

    source = FakeSource()
    with patch("typoon.translation.translate.translate_pages", mock_translate_pages):
        await run_pipeline(engine, store, config, 1, [(1, source)], hook=hook)

    event_names = [e.name for e in hook.events]
    results = {
        "has_translation_ready": "TranslationReady" in event_names,
        "event_order": event_names,
    }

    if "TranslationReady" in event_names:
        tr_idx = event_names.index("TranslationReady")
        erase_indices = [i for i, n in enumerate(event_names) if n == "PageErased"]
        results["ready_before_erase"] = all(tr_idx < ei for ei in erase_indices) if erase_indices else True
    else:
        results["ready_before_erase"] = False

    return results


# ── Benchmark 5: Preprocess API ──────────────────────────────────────

async def bench_preprocess_api() -> dict:
    """Verify Engine.preprocess() returns (list[Page], ChapterImages)."""
    engine = _make_engine()
    source = FakeSource()
    hook = RecorderHook()

    pages, images = engine.preprocess(source, hook)

    results = {
        "returns_pages": isinstance(pages, list) and all(isinstance(p, Page) for p in pages),
        "returns_chapter_images": isinstance(images, ChapterImages),
        "page_count_match": len(pages) == source.page_count(),
        "images_page_count": images.page_count() == source.page_count(),
        "images_alive": images.alive,
        "bubbles_detected": all(len(p.bubbles) > 0 for p in pages),
    }

    # Verify free() works
    images.free()
    results["images_freed"] = not images.alive

    return results


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Pipeline Benchmark — Simulated Delays")
    print(f"  Chapters: {NUM_CHAPTERS}")
    print(f"  Scan: {SCAN_DELAY*1000:.0f}ms  Translate: {TRANSLATE_DELAY*1000:.0f}ms  "
          f"Erase: {ERASE_DELAY*1000:.0f}ms  Knowledge: {KNOWLEDGE_DELAY*1000:.0f}ms")
    per_chapter = SCAN_DELAY + TRANSLATE_DELAY + ERASE_DELAY + KNOWLEDGE_DELAY
    print(f"  Per-chapter total: {per_chapter*1000:.0f}ms")
    print("=" * 60)

    # 1. Sequential baseline
    print("\n▸ Sequential (baseline)...")
    seq_time, seq_hook = await bench_sequential(NUM_CHAPTERS)
    print(f"  Total: {seq_time*1000:.0f}ms")

    # 2. Queue-based pipeline
    print("\n▸ Queue pipeline (scan N+1 during translate N)...")
    la_time, la_hook = await bench_queue_pipeline(NUM_CHAPTERS)
    print(f"  Total: {la_time*1000:.0f}ms")

    savings = seq_time - la_time
    pct = (savings / seq_time) * 100 if seq_time > 0 else 0
    print(f"\n  ⚡ Saved: {savings*1000:.0f}ms ({pct:.1f}%)")
    theoretical_savings = SCAN_DELAY * (NUM_CHAPTERS - 1)
    print(f"  Theoretical max savings: {theoretical_savings*1000:.0f}ms "
          f"(scan overlaps with translate for {NUM_CHAPTERS-1} chapters)")

    # 3. Model lifecycle
    print("\n▸ Model lifecycle...")
    lifecycle = await bench_model_lifecycle()
    for k, v in lifecycle.items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}")

    # 4. Progressive output
    print("\n▸ Progressive output...")
    progressive = await bench_progressive_output()
    for k, v in progressive.items():
        if k == "event_order":
            print(f"  Event flow: {' → '.join(v)}")
        else:
            status = "✓" if v else "✗"
            print(f"  {status} {k}")

    # 5. Preprocess API
    print("\n▸ Preprocess API...")
    preprocess = await bench_preprocess_api()
    for k, v in preprocess.items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}")

    # Timeline for queue pipeline
    print("\n▸ Queue pipeline timeline:")
    for ev in la_hook.events:
        print(f"  [{ev.ts*1000:6.0f}ms] {ev.name:20s} {ev.detail}")

    print("\n" + "=" * 60)
    all_pass = (
        pct > 0
        and all(lifecycle.values())
        and progressive["has_translation_ready"]
        and progressive["ready_before_erase"]
        and all(v for k, v in preprocess.items())
    )
    print(f"Result: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
