"""Shared fixtures for vision pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from typoon.vision.types import TextMask, TextRegion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def models_available(*names: str) -> bool:
    return all((MODELS_DIR / n).exists() for n in names)


skip_no_ppocr_det = pytest.mark.skipif(
    not models_available("ppocr-det.safetensors", "ppocr-det-config.json"),
    reason="ppocr-det.safetensors not found",
)
skip_no_lama = pytest.mark.skipif(
    not models_available("lama-manga.safetensors"),
    reason="lama-manga.safetensors not found",
)

_DUMMY_CROP = np.zeros((1, 1, 3), dtype=np.uint8)


def make_line(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> TextRegion:
    return TextRegion(
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        crop=_DUMMY_CROP,
        confidence=conf,
        mask=None,
    )


def make_mask(x: int, y: int, w: int, h: int, fill: int = 255) -> TextMask:
    img = np.full((h, w), fill, dtype=np.uint8)
    return TextMask(x=x, y=y, image=img)


# ── LLM + Translation shared fixtures ────────────────────────────

from typoon.llm.ir import CallResponse, Message, ToolDef, ToolCallMsg, ToolResponse
from typoon.runs.events import Hook
from typoon.adapters.ctx import TranslateCtx
from typoon.domain.prepared import Chapter, Page as PreparedPage
from typoon.domain.scan import Box, Bubble as ScannedBubble, Chapter as ScannedChapter, Page as ScannedPage


class MockProvider:
    """Returns pre-programmed LLM responses."""
    def __init__(self, responses: list[CallResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.call_count = 0

    async def call(self, messages, tools) -> CallResponse:
        self.call_count += 1
        return self._responses.pop(0) if self._responses else CallResponse()


class MockStore:
    """In-memory store for testing."""
    def __init__(self):
        self.briefs = {}
        self.glossary = {}
        self._translations = {}

    async def get_project(self, pid): return {"source_lang": "en", "target_lang": "vi"}
    async def get_project_by_title(self, title): return None
    async def get_project_by_url(self, url): return None
    async def get_glossary(self, pid): return dict(self.glossary)
    async def save_chapter_brief(self, chapter_id, brief): self.briefs[chapter_id] = brief
    async def get_chapter_brief(self, chapter_id): return self.briefs.get(chapter_id)
    async def get_recent_chapter_briefs(self, pid, before_chapter_idx, limit=3):
        rows = []
        for ch_id, brief in sorted(self.briefs.items(), reverse=True):
            rows.append({"chapter": 0, "brief": brief, "summary": brief.get("summary", "")})
        return rows[:limit]
    async def search_briefs(self, pid, queries, limit=10, *, before_chapter_idx=None): return []
    async def save_translations(self, chapter_id, records): self._translations[chapter_id] = records
    async def save_bubbles(self, chapter_id, bubbles): pass
    async def get_bubbles(self, chapter_id): return []
    async def get_translations(self, chapter_id): return {}
    async def get_chapter_translations(self, pid, ch): return []
    async def glossary_search(self, pid, q): return []
    async def glossary_upsert(self, pid, s, t, n): self.glossary[s] = t
    async def search_context(self, pid, q, scope, limit): return []
    async def get_chapter_pairs(self, pid, ch): return []
    async def get_chapter_status(self, pid, idx): return None
    async def get_chapter_retry_count(self, pid, idx): return 0
    async def increment_retry_count(self, pid, idx): pass
    async def delete_chapter_data(self, pid, idx): pass
    async def add_chapter(self, pid, idx, **kw): pass
    async def set_chapter_status(self, pid, idx, status): pass


_POLY = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
_BOX = Box(
    polygon=_POLY,
    fit=[0, 0, 100, 50],
    erase=[0, 0, 100, 50],
    text=[5, 5, 95, 45],
)
_PREPARED = Chapter(
    source="test",
    pages=(PreparedPage(index=0, width=800, height=1200),),
)


class FakePreparedReader:
    """In-memory PreparedReader stand-in for tests that don't decode pixels.

    The translate/context paths only call read_rgb when build_chapter_brief
    requests visual context for address-sensitive bubbles. Tests using
    MockProvider never trigger that path, so a stub is enough.
    """

    def __init__(self, pages: tuple[PreparedPage, ...] = _PREPARED.pages) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def chapter(self, source: str = "") -> Chapter:
        return Chapter(source=source, pages=self._pages)

    def read_rgb(self, index: int) -> np.ndarray:
        page = self._pages[index]
        return np.zeros((page.height, page.width, 3), dtype=np.uint8)

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakePreparedReader":
        return self

    def __exit__(self, *_) -> None:
        pass


def make_scanned_chapter(n_bubbles: int = 3) -> ScannedChapter:
    bubbles = tuple(
        ScannedBubble(
            idx=i, page_index=0,
            source_text=f"text_{i}",
            confidence=0.9,
            box=_BOX,
        )
        for i in range(n_bubbles)
    )
    page = ScannedPage(index=0, width=800, height=1200, bubbles=bubbles)
    return ScannedChapter(prepared=_PREPARED, pages=(page,))


def make_session(
    n_bubbles: int = 3,
    provider_responses: list[CallResponse] | None = None,
    glossary: dict[str, str] | None = None,
) -> tuple[ScannedChapter, FakePreparedReader, TranslateCtx]:
    """Create a ScannedChapter + PreparedReader stub + TranslateCtx."""
    scanned = make_scanned_chapter(n_bubbles)
    reader  = FakePreparedReader()
    provider = MockProvider(provider_responses)
    ctx = TranslateCtx(
        translation_provider=provider,
        context_provider=provider,
        vision_provider=provider,
        store=MockStore(),
        project_id=1,
        chapter_id=1,
        chapter_idx=1.0,
        source_lang="en",
        target_lang="vi",
        hook=Hook(),
    )
    return scanned, reader, ctx


def make_text_response(text: str) -> CallResponse:
    """Build a plain-text response (pass 1)."""
    return CallResponse(text=text)
