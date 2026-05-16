"""Lens detector — row gap recovery + language-hint tests.

The detector handles two recoveries on top of raw Lens output:

  * `_suspicious_line_indices` flags non-edge rows whose width is well
    below the median of the block's other rows. Lens drops glyphs
    around dense decoration runs (e.g. CJK ellipsis 「······」) and
    emits only the unaffected suffix on that row.
  * `_recover_row_gaps` re-OCRs the suspicious row with a tight crop
    and splices the recovered text back into the block.

Re-OCR uses a stubbed Lens API; no network.
"""

from __future__ import annotations

import asyncio

import numpy as np

from typoon.vision.contracts import LineBox, TextBlock, WordBox
from typoon.vision.detectors.lens_blocks import (
    _lens_lang_hint,
    _recover_row_gaps,
    _suspicious_line_indices,
)


def _block(bbox, text, *, lines=(), words=()):
    return TextBlock(
        bbox=bbox, polygon=None, confidence=1.0,
        text=text, detector="lens_blocks",
        lines=tuple(lines), words=tuple(words),
    )


# ─── _lens_lang_hint ─────────────────────────────────────────────────────


def test_lens_lang_hint_unset_returns_auto():
    assert _lens_lang_hint(None) == ""
    assert _lens_lang_hint("") == ""


def test_lens_lang_hint_english_returns_auto():
    """English source keeps auto-detect so mixed-script SFX survive."""
    assert _lens_lang_hint("en") == ""
    assert _lens_lang_hint("en-US") == ""
    assert _lens_lang_hint("EN") == ""


def test_lens_lang_hint_japanese():
    assert _lens_lang_hint("ja") == "ja"
    assert _lens_lang_hint("ja-JP") == "ja"


def test_lens_lang_hint_chinese_variants():
    assert _lens_lang_hint("zh") == "zh-Hans"
    assert _lens_lang_hint("zh-CN") == "zh-Hans"
    assert _lens_lang_hint("zh-Hans") == "zh-Hans"
    assert _lens_lang_hint("zh-Hant") == "zh-Hant"
    assert _lens_lang_hint("zh-TW") == "zh-Hant"


def test_lens_lang_hint_unknown_falls_to_auto():
    assert _lens_lang_hint("th") == ""
    assert _lens_lang_hint("hi") == ""


# ─── _suspicious_line_indices ────────────────────────────────────────────


def _line(x1, y1, x2, y2, text):
    return LineBox(bbox=(x1, y1, x2, y2), text=text)


def test_no_suspicious_lines_when_widths_uniform():
    block = _block(
        (100, 100, 300, 200), "row1 row2 row3",
        lines=(
            _line(100, 100, 280, 130, "row1"),
            _line(100, 140, 290, 170, "row2"),
            _line(100, 180, 285, 200, "row3"),
        ),
    )
    assert _suspicious_line_indices(block) == []


def test_no_suspicious_lines_when_fewer_than_3():
    """Need at least 3 lines to compute a meaningful median."""
    block = _block(
        (100, 100, 300, 170), "short long",
        lines=(
            _line(100, 100, 130, 130, "short"),
            _line(100, 140, 290, 170, "long"),
        ),
    )
    assert _suspicious_line_indices(block) == []


def test_short_middle_row_flagged():
    """L[2] is 25% of median (others 200, 220, 210 -> median 210)."""
    block = _block(
        (100, 100, 320, 260), "a b c d",
        lines=(
            _line(100, 100, 300, 130, "row 0"),
            _line(100, 140, 320, 170, "row 1"),
            _line(100, 180, 150, 210, "x"),         # short!
            _line(100, 220, 310, 260, "row 3"),
        ),
    )
    assert _suspicious_line_indices(block) == [2]


def test_short_edge_row_not_flagged():
    """First / last rows are often legitimately short — ragged justification."""
    block = _block(
        (100, 100, 320, 220), "a b c",
        lines=(
            _line(100, 100, 130, 130, "x"),         # short but first
            _line(100, 140, 320, 170, "row 1"),
            _line(100, 180, 130, 220, "x"),         # short but last
        ),
    )
    assert _suspicious_line_indices(block) == []


def test_multiple_short_rows_flagged():
    block = _block(
        (100, 100, 320, 300), "...",
        lines=(
            _line(100, 100, 300, 130, "wide 0"),
            _line(100, 140, 150, 170, "x"),         # short
            _line(100, 180, 310, 220, "wide 2"),
            _line(100, 230, 150, 260, "x"),         # short
            _line(100, 270, 320, 300, "wide 4"),
        ),
    )
    assert _suspicious_line_indices(block) == [1, 3]


# ─── _recover_row_gaps ───────────────────────────────────────────────────


class _StubLensApi:
    """Stand-in for chrome_lens_py.LensAPI used inside `_reocr_row`.

    Records ocr_language calls and returns scripted paragraph text.
    """
    def __init__(self, scripted_texts: list[str]):
        self._scripted = list(scripted_texts)
        self.calls: list[dict] = []

    async def process_image(self, image, *, ocr_language, output_format):
        self.calls.append({
            "shape": image.shape,
            "ocr_language": ocr_language,
            "output_format": output_format,
        })
        text = self._scripted.pop(0) if self._scripted else ""
        return {
            "detailed_blocks": [
                {"text": text, "geometry": None}
            ] if text else []
        }


def test_recover_row_gaps_replaces_short_row_text():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    block = _block(
        (50, 50, 350, 250), "row 0\nrow 1\nx\nrow 3",
        lines=(
            _line(50, 50, 300, 90,  "row 0"),
            _line(50, 100, 320, 140, "row 1"),
            _line(50, 150, 100, 190, "x"),           # suspicious — width 50 / median 270+
            _line(50, 200, 310, 250, "row 3"),
        ),
    )
    api = _StubLensApi(scripted_texts=["recovered glyphs here"])
    out = asyncio.run(_recover_row_gaps(api, image, [block], lang_hint="zh-Hans"))
    assert len(out) == 1
    recovered = out[0]
    # Block text reassembled with the recovered row
    assert "recovered glyphs here" in recovered.text
    assert recovered.lines[2].text == "recovered glyphs here"
    # Other lines preserved
    assert recovered.lines[0].text == "row 0"
    assert recovered.lines[1].text == "row 1"
    assert recovered.lines[3].text == "row 3"
    # bbox and geometry intact
    assert recovered.bbox == block.bbox
    assert recovered.lines[2].bbox == block.lines[2].bbox
    # Lens called with the right language
    assert len(api.calls) == 1
    assert api.calls[0]["ocr_language"] == "zh-Hans"


def test_recover_row_gaps_passes_lang_hint():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    block = _block(
        (50, 50, 350, 250), "...",
        lines=(
            _line(50, 50, 300, 90,  "row 0"),
            _line(50, 100, 320, 140, "row 1"),
            _line(50, 150, 100, 190, "x"),
            _line(50, 200, 310, 250, "row 3"),
        ),
    )
    api = _StubLensApi(scripted_texts=["new"])
    asyncio.run(_recover_row_gaps(api, image, [block], lang_hint="ja"))
    assert api.calls[0]["ocr_language"] == "ja"


def test_recover_row_gaps_noop_when_no_suspicious_rows():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    block = _block(
        (50, 50, 350, 250), "uniform",
        lines=(
            _line(50, 50, 300, 90,  "row 0"),
            _line(50, 100, 290, 140, "row 1"),
            _line(50, 150, 310, 190, "row 2"),
        ),
    )
    api = _StubLensApi(scripted_texts=[])
    out = asyncio.run(_recover_row_gaps(api, image, [block], lang_hint=""))
    assert out[0] is block  # exact same instance returned, no rebuild
    assert api.calls == []  # no Lens calls at all


def test_recover_row_gaps_keeps_block_when_reocr_empty():
    """Lens returning empty = no text recovered. Keep the original block."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    block = _block(
        (50, 50, 350, 250), "...",
        lines=(
            _line(50, 50, 300, 90,  "row 0"),
            _line(50, 100, 320, 140, "row 1"),
            _line(50, 150, 100, 190, "keep me"),
            _line(50, 200, 310, 250, "row 3"),
        ),
    )
    api = _StubLensApi(scripted_texts=[""])
    out = asyncio.run(_recover_row_gaps(api, image, [block], lang_hint=""))
    # No mutation: returned block is the original instance
    assert out[0] is block
    assert out[0].lines[2].text == "keep me"
