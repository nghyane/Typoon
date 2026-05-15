"""render_chapter erase-mask filtering tests.

The fix locked here: bubbles flagged `kind="skip"` (logos, page numbers,
fan-group credits, watermarks that the LLM declined to translate) must
NOT have their erase masks applied. Erasing them would wipe artwork the
user explicitly wanted preserved.

Tests exercise `_is_renderable` (the single predicate) and the gather
comprehensions that consume it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from typoon.adapters.mask_store import BubbleMasks
from typoon.domain import scan, translate
from typoon.stages.render import _is_renderable
from typoon.vision.contracts import TextMask


# ─── Fakes ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _RecordingEraser:
    """Stand-in for VisionRuntime.eraser. Captures every mask passed."""
    name: str = "recording"
    seen:  list[TextMask] = None  # set in __post_init__

    def __post_init__(self):
        self.seen = []

    async def erase(self, canvas: np.ndarray, masks):
        self.seen.extend(masks)
        return canvas


# ─── Helpers ──────────────────────────────────────────────────────────────


def _box(x: int) -> scan.Box:
    return scan.Box(
        polygon=[[float(x), 0], [float(x + 10), 0], [float(x + 10), 10], [float(x), 10]],
        fit=[x, 0, x + 10, 10],
        erase=[x, 0, x + 10, 10],
        text=[x, 0, x + 10, 10],
    )


def _bubble(
    idx: int, source_text: str, *, kind: str, translated: str,
) -> translate.Bubble:
    src = scan.Bubble(
        idx=idx,
        page_index=0,
        source_text=source_text,
        confidence=1.0,
        box=_box(idx * 20),
    )
    return translate.Bubble(
        source=src,
        translation_key=f"K{idx}",
        translated_text=translated,
        kind=kind,
    )


def _mask(idx: int) -> TextMask:
    """Tagged-by-index mask for assertion clarity."""
    img = np.full((10, 10), 255, dtype=np.uint8)
    return TextMask(x=idx * 20, y=0, image=img)


# ─── Tests ────────────────────────────────────────────────────────────────


def test_erase_filter_skips_kind_skip_bubbles():
    """Reproduce render's erase-mask gathering and assert skip masks
    are excluded. Uses _is_renderable directly so the predicate change
    is detected by the test (the comprehension itself is trivial).
    """
    bubbles = (
        _bubble(0, "DIALOGUE",         kind="dialogue", translated="Lời thoại"),
        _bubble(1, "LOGO/CREDIT",      kind="skip",     translated=""),
        _bubble(2, "PAGE NUMBER",      kind="skip",     translated=""),
        _bubble(3, "ANOTHER DIALOGUE", kind="dialogue", translated="Tiếp"),
    )
    page_masks: dict[int, BubbleMasks] = {
        i: BubbleMasks(erase_masks=(_mask(i),), text_masks=(_mask(i),))
        for i in range(4)
    }

    renderable = [tb for tb in bubbles if _is_renderable(tb)]
    erase_masks = tuple(
        m
        for tb in renderable
        for bm in [page_masks.get(tb.idx)]
        if bm is not None
        for m in bm.erase_masks
    )

    assert len(erase_masks) == 2
    seen_x = sorted(m.x for m in erase_masks)
    assert seen_x == [0, 60]


def test_erase_filter_keeps_all_when_none_skipped():
    bubbles = (
        _bubble(0, "A", kind="dialogue", translated="A"),
        _bubble(1, "B", kind="sfx",      translated="B"),
    )
    page_masks = {
        i: BubbleMasks(erase_masks=(_mask(i),), text_masks=(_mask(i),))
        for i in range(2)
    }
    renderable = [tb for tb in bubbles if _is_renderable(tb)]
    erase_masks = tuple(
        m
        for tb in renderable
        for bm in [page_masks.get(tb.idx)]
        if bm is not None
        for m in bm.erase_masks
    )
    assert len(erase_masks) == 2


def test_erase_filter_skips_all_when_every_bubble_skipped():
    bubbles = (
        _bubble(0, "LOGO",   kind="skip", translated=""),
        _bubble(1, "CREDIT", kind="skip", translated=""),
    )
    page_masks = {
        i: BubbleMasks(erase_masks=(_mask(i),), text_masks=(_mask(i),))
        for i in range(2)
    }
    renderable = [tb for tb in bubbles if _is_renderable(tb)]
    erase_masks = tuple(
        m
        for tb in renderable
        for bm in [page_masks.get(tb.idx)]
        if bm is not None
        for m in bm.erase_masks
    )
    assert erase_masks == ()


def test_is_renderable_predicate_rejects_only_skip():
    """The single predicate used by both gather sites in render."""
    assert _is_renderable(_bubble(0, "x", kind="dialogue", translated="y"))
    assert _is_renderable(_bubble(0, "x", kind="sfx",      translated="y"))
    # Even empty translated_text doesn't trigger skip — that's a separate
    # check at the call site (active_triples requires non-empty text).
    assert _is_renderable(_bubble(0, "x", kind="dialogue", translated=""))
    assert not _is_renderable(_bubble(0, "x", kind="skip", translated=""))


@pytest.mark.asyncio
async def test_recording_eraser_only_receives_non_skip_masks():
    bubbles = (
        _bubble(0, "DIALOGUE", kind="dialogue", translated="OK"),
        _bubble(1, "LOGO",     kind="skip",     translated=""),
    )
    page_masks = {
        i: BubbleMasks(erase_masks=(_mask(i),), text_masks=(_mask(i),))
        for i in range(2)
    }
    eraser = _RecordingEraser()

    renderable = [tb for tb in bubbles if _is_renderable(tb)]
    erase_masks = tuple(
        m
        for tb in renderable
        for bm in [page_masks.get(tb.idx)]
        if bm is not None
        for m in bm.erase_masks
    )
    canvas = np.zeros((100, 100, 4), dtype=np.uint8)
    await eraser.erase(canvas, erase_masks)

    assert len(eraser.seen) == 1
    assert eraser.seen[0].x == 0
