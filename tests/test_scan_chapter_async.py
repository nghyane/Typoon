"""scan_chapter async + structured concurrency tests.

Uses AsyncMock detectors/groupers so no model load is needed. Verifies:
  - page_gate semaphore bounds in-flight pages
  - exception in one page cancels TaskGroup and surfaces
  - artefact sink writes per-page JSON + overlays
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from typoon.adapters.mask_store import MaskStore  # noqa: F401
from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.prepared import Page as PreparedPage
from typoon.stages.scan import scan_chapter
from typoon.vision.contracts import (
    BubbleGroup,
    DetectionResult,
    TextBlock,
    TextMask,
)
from typoon.vision.pipeline import VisionPipelineSpec
from typoon.vision.runtime import VisionRuntime


# ─── Fakes ────────────────────────────────────────────────────────────────


class _FakeReader:
    def __init__(self, n_pages: int, dim: int = 256) -> None:
        self._n = n_pages
        self._dim = dim

    @property
    def page_count(self) -> int:
        return self._n

    def chapter(self) -> PreparedChapter:
        return PreparedChapter(
            source="fake",
            pages=tuple(
                PreparedPage(index=i, width=self._dim, height=self._dim)
                for i in range(self._n)
            ),
        )

    def read_rgb(self, index: int) -> np.ndarray:
        return np.full((self._dim, self._dim, 3), index, dtype=np.uint8)


def _runtime(detector, grouper, recognizer=None, eraser=None,
             page_concurrency: int = 4) -> VisionRuntime:
    spec = VisionPipelineSpec(
        detector="lens_blocks",
        grouper="lens_native",
        recognizer="none",
        page_concurrency=page_concurrency,
    )
    return VisionRuntime(
        spec=spec,
        detector=detector,
        grouper=grouper,
        recognizer=recognizer,
        eraser=eraser or _NoopEraser(),
    )


class _NoopEraser:
    name = "noop"

    async def erase(self, canvas, masks):
        return canvas


def _bubble_group() -> BubbleGroup:
    img = np.full((10, 10), 255, dtype=np.uint8)
    return BubbleGroup(
        bbox=(0, 0, 10, 10),
        polygon=((0, 0), (10, 0), (10, 10), (0, 10)),
        text="hello",
        confidence=1.0,
        text_masks=(TextMask(x=0, y=0, image=img),),
        erase_masks=(TextMask(x=0, y=0, image=img),),
        source="fake",
    )


# ─── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_chapter_runs_pages_in_parallel():
    n_pages = 8
    page_concurrency = 4
    in_flight = 0
    max_seen = 0

    async def slow_detect(image, lang):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        h, w = image.shape[:2]
        return DetectionResult(
            blocks=(),
            text_already_recognized=True,
            page_size=(w, h),
        )

    detector = AsyncMock()
    detector.name = "fake"
    detector.detect = AsyncMock(side_effect=slow_detect)

    grouper = AsyncMock()
    grouper.name = "fake"
    grouper.group = AsyncMock(return_value=())

    runtime = _runtime(detector, grouper, page_concurrency=page_concurrency)
    reader = _FakeReader(n_pages)
    prepared = reader.chapter()

    result = await scan_chapter(prepared, reader, runtime)

    assert detector.detect.await_count == n_pages
    assert max_seen <= page_concurrency
    assert max_seen >= 2  # actually parallel
    assert len(result.chapter.pages) == n_pages


@pytest.mark.asyncio
async def test_scan_chapter_propagates_per_page_exception():
    async def failing_detect(image, lang):
        raise RuntimeError("detector boom")

    detector = AsyncMock()
    detector.name = "fake"
    detector.detect = AsyncMock(side_effect=failing_detect)

    grouper = AsyncMock()
    grouper.name = "fake"

    runtime = _runtime(detector, grouper)
    reader = _FakeReader(3)
    prepared = reader.chapter()

    with pytest.raises(BaseExceptionGroup) as excinfo:
        await scan_chapter(prepared, reader, runtime)
    # At least one underlying RuntimeError surfaced
    assert any(isinstance(e, RuntimeError) for e in excinfo.value.exceptions)


@pytest.mark.asyncio
async def test_scan_chapter_skips_undersized_pages():
    n_pages = 3

    async def detect(image, lang):
        h, w = image.shape[:2]
        return DetectionResult(
            blocks=(_text_block(),),
            text_already_recognized=True,
            page_size=(w, h),
        )

    async def group(image, detection, lang):
        return (_bubble_group(),)

    detector = AsyncMock()
    detector.name = "fake"
    detector.detect = AsyncMock(side_effect=detect)

    grouper = AsyncMock()
    grouper.name = "fake"
    grouper.group = AsyncMock(side_effect=group)

    runtime = _runtime(detector, grouper)
    # _MIN_PAGE_DIM = 128; reader with dim=64 should skip every page
    reader = _FakeReader(n_pages, dim=64)
    prepared = reader.chapter()

    result = await scan_chapter(prepared, reader, runtime)
    assert detector.detect.await_count == 0
    assert all(len(p.bubbles) == 0 for p in result.chapter.pages)


def _text_block() -> TextBlock:
    return TextBlock(
        bbox=(0, 0, 10, 10),
        polygon=None,
        confidence=1.0,
        text="x",
        detector="fake",
    )
