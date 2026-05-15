"""scan_chapter language sanity checks."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.prepared import Page as PreparedPage
from typoon.stages.scan import (
    _normalize_lang,
    _script_of,
    _scripts_differ,
    _vote_lang,
    scan_chapter,
)
from typoon.vision.contracts import (
    BubbleGroup,
    DetectionResult,
    TextBlock,
    TextMask,
)
from typoon.vision.pipeline import VisionPipelineSpec
from typoon.vision.runtime import VisionRuntime


def test_normalize_lang_strips_region():
    assert _normalize_lang("zh-CN") == "zh"
    assert _normalize_lang("en-US") == "en"
    assert _normalize_lang("ja") == "ja"
    assert _normalize_lang("EN") == "en"


def test_script_of_groups_by_writing_system():
    assert _script_of("en") == "latin"
    assert _script_of("es-la") == "latin"
    assert _script_of("vi") == "latin"
    assert _script_of("ja") == "japanese"
    assert _script_of("ko") == "korean"
    assert _script_of("zh-CN") == "han"
    assert _script_of("ru") == "cyrillic"
    assert _script_of("ar") == "arabic"


def test_scripts_differ_only_across_writing_systems():
    # Same script: should NOT differ — Lens often confuses these
    assert not _scripts_differ("es-la", "en")
    assert not _scripts_differ("fr", "vi")
    assert not _scripts_differ("pt", "en")
    # Different script: SHOULD differ — likely wrong chapter language
    assert _scripts_differ("ja", "en")
    assert _scripts_differ("ko", "zh")
    assert _scripts_differ("ru", "en")


def test_vote_lang_picks_majority():
    assert _vote_lang(["en", "en", "ja", "en"]) == "en"
    assert _vote_lang(["ja", "ja"]) == "ja"
    assert _vote_lang([]) is None
    assert _vote_lang([None, None]) is None
    # Tie: most_common is order-stable on Python 3.7+
    assert _vote_lang(["en", "ja"]) in {"en", "ja"}


# ─── Mismatch warning ─────────────────────────────────────────────────────


class _FakeReader:
    def __init__(self, n: int = 2, dim: int = 256) -> None:
        self._n = n
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


class _FakeDetector:
    name = "fake"

    def __init__(self, lang: str) -> None:
        self._lang = lang
        self.calls = 0

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        self.calls += 1
        h, w = image.shape[:2]
        block = TextBlock(
            bbox=(10, 10, 100, 50),
            polygon=None,
            confidence=1.0,
            text="hello",
            detector=self.name,
        )
        return DetectionResult(
            blocks=(block,),
            text_already_recognized=True,
            page_size=(w, h),
            detected_lang=self._lang,
        )


class _FakeGrouper:
    name = "fake"

    async def group(self, image, detection, lang):
        img = np.full((10, 10), 255, dtype=np.uint8)
        return tuple(
            BubbleGroup(
                bbox=b.bbox,
                polygon=((b.bbox[0], b.bbox[1]), (b.bbox[2], b.bbox[1]),
                         (b.bbox[2], b.bbox[3]), (b.bbox[0], b.bbox[3])),
                text=b.text or "",
                confidence=1.0,
                text_masks=(TextMask(x=b.bbox[0], y=b.bbox[1], image=img),),
                erase_masks=(TextMask(x=b.bbox[0], y=b.bbox[1], image=img),),
                source="fake",
            )
            for b in detection.blocks
        )


class _NoopEraser:
    name = "noop"

    async def erase(self, canvas, masks):
        return canvas


def _runtime(detector_lang: str) -> VisionRuntime:
    spec = VisionPipelineSpec(
        detector="lens_blocks",
        grouper="lens_native",
        recognizer="none",
        page_concurrency=2,
    )
    return VisionRuntime(
        spec=spec,
        detector=_FakeDetector(detector_lang),
        grouper=_FakeGrouper(),
        recognizer=None,
        eraser=_NoopEraser(),
    )


@pytest.mark.asyncio
async def test_scan_chapter_warns_on_script_mismatch(caplog):
    """source_lang='ja' (Japanese script) vs detected='en' (Latin) → warn."""
    reader = _FakeReader(2)
    runtime = _runtime(detector_lang="en")
    with caplog.at_level(logging.WARNING, logger="typoon.stages.scan"):
        result = await scan_chapter(
            reader.chapter(), reader, runtime, source_lang="ja",
        )
    assert result.detected_lang == "en"
    assert any(
        "source_lang='ja'" in rec.message and "'en'" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_scan_chapter_silent_within_same_script(caplog):
    """Spanish chapter where Lens reports 'en' should NOT warn — both Latin."""
    reader = _FakeReader(2)
    runtime = _runtime(detector_lang="en")
    with caplog.at_level(logging.WARNING, logger="typoon.stages.scan"):
        result = await scan_chapter(
            reader.chapter(), reader, runtime, source_lang="es-la",
        )
    assert result.detected_lang == "en"
    assert not any("source_lang" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_scan_chapter_silent_when_lang_matches(caplog):
    reader = _FakeReader(2)
    runtime = _runtime(detector_lang="en")
    with caplog.at_level(logging.WARNING, logger="typoon.stages.scan"):
        result = await scan_chapter(
            reader.chapter(), reader, runtime, source_lang="en",
        )
    assert result.detected_lang == "en"
    assert not any("source_lang" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_scan_chapter_silent_when_region_only_differs(caplog):
    """source_lang='zh-CN' vs detected_lang='zh' should NOT warn."""
    reader = _FakeReader(2)
    runtime = _runtime(detector_lang="zh")
    with caplog.at_level(logging.WARNING, logger="typoon.stages.scan"):
        await scan_chapter(
            reader.chapter(), reader, runtime, source_lang="zh-CN",
        )
    assert not any("source_lang" in r.message for r in caplog.records)
