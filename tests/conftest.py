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
from typoon.events import Hook
from typoon.domain.bubble import Bubble, Page, Session


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
        self.snapshots = {}
        self.notes = []
        self.glossary = {}
        self._translations = {}

    async def get_project(self, pid): return {"source_lang": "en", "target_lang": "vi"}
    async def get_project_by_title(self, title): return None
    async def get_project_by_url(self, url): return None
    async def get_glossary(self, pid): return dict(self.glossary)
    async def get_knowledge(self, pid, before_chapter):
        for ch in sorted(self.snapshots, reverse=True):
            if ch < before_chapter: return self.snapshots[ch]
        return None
    async def save_knowledge(self, pid, ch, snap): self.snapshots[ch] = snap
    async def save_translations(self, pid, ch, bubbles): self._translations[(pid, ch)] = bubbles
    async def get_chapter_translations(self, pid, ch): return []
    async def glossary_search(self, pid, q): return []
    async def glossary_upsert(self, pid, s, t, n): self.glossary[s] = t
    async def add_note(self, pid, ch, nt, c): self.notes.append({"chapter": ch, "type": nt, "content": c})
    async def search_context(self, pid, q, scope, limit): return []
    async def get_chapter_pairs(self, pid, ch): return []
    async def get_chapter_status(self, pid, idx): return None
    async def get_chapter_retry_count(self, pid, idx): return 0
    async def increment_retry_count(self, pid, idx): pass
    async def delete_chapter_data(self, pid, idx): pass
    async def add_chapter(self, pid, idx, **kw): pass
    async def set_chapter_status(self, pid, idx, status): pass


class MockSource:
    """Fake chapter source."""
    def __init__(self, page_count: int = 0):
        self._page_count = page_count
    def page_count(self): return self._page_count
    def load_page(self, index): return np.zeros((100, 100, 3), dtype=np.uint8)
    async def fetch(self): pass


def make_session(
    n_bubbles: int = 3,
    provider_responses: list[CallResponse] | None = None,
    glossary: dict[str, str] | None = None,
) -> tuple[list[Page], Session]:
    """Create test pages + session with mock provider."""
    bubbles = [
        Bubble(idx=i, page_index=0, source_text=f"text_{i}",
               polygon=[[0, 0], [100, 0], [100, 50], [0, 50]])
        for i in range(n_bubbles)
    ]
    pages = [Page(index=0, bubbles=bubbles)]
    provider = MockProvider(provider_responses)
    session = Session(
        store=MockStore(), source=MockSource(), project_id=1,
        source_lang="en", target_lang="vi",
        provider=provider, context_provider=provider, hook=Hook(),
        glossary=glossary or {},
    )
    return pages, session


def make_translate_response(items: list[tuple[str, str]]) -> CallResponse:
    """Build a translate tool call response."""
    import json
    translations = [{"id": bid, "translated_text": text} for bid, text in items]
    return CallResponse(tool_calls=[ToolCallMsg(
        id="c1", name="translate",
        arguments=json.dumps({"translations": translations}),
    )])
