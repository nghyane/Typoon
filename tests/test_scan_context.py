"""Tests for storyboard layout + scan_context parser."""

from __future__ import annotations

import numpy as np
import pytest

from typoon.domain.scan import Box, Bubble, BubbleKey
from typoon.stages.brief import Character, ChapterBrief, brief_slice
from typoon.stages.scan_context import (
    _parse_kv,
    _parse_reply,
    _full_noise_pages,
)
from typoon.stages.storyboard import chunk_pages


_BOX = Box(
    polygon=[[10.0, 10.0], [100.0, 10.0], [100.0, 80.0], [10.0, 80.0]],
    fit=[10, 10, 100, 80],
    erase=[10, 10, 100, 80],
    text=[10, 10, 100, 80],
)


def _bk(key: str, idx: int, text: str, page: int = 0) -> BubbleKey:
    return BubbleKey(
        key=key,
        bubble=Bubble(
            idx=idx, page_index=page, source_text=text,
            confidence=0.9, box=_BOX,
        ),
    )


class TestChunking:
    @pytest.mark.parametrize("n,expected", [
        (0, []),
        (1, [range(0, 1)]),
        (4, [range(0, 4)]),
        (9, [range(0, 9)]),
        (10, [range(0, 9), range(9, 10)]),
        (20, [range(0, 9), range(9, 18), range(18, 20)]),
    ])
    def test_split(self, n, expected):
        assert chunk_pages(n) == expected


class TestKVParser:
    def test_basic_pairs(self):
        out = _parse_kv("name=Denji gender=male")
        assert out == {"name": "Denji", "gender": "male"}

    def test_quoted_value_with_spaces(self):
        out = _parse_kv('name="white-haired girl" gender=female')
        assert out["name"] == "white-haired girl"
        assert out["gender"] == "female"

    def test_quoted_value_with_commas(self):
        out = _parse_kv('mood="bleak, hungry, desperate" register=casual')
        assert out["mood"] == "bleak, hungry, desperate"
        assert out["register"] == "casual"

    def test_unterminated_quote_consumes_rest(self):
        out = _parse_kv('name="unclosed')
        assert out["name"] == "unclosed"

    def test_empty_string(self):
        assert _parse_kv("") == {}


class TestReplyParser:
    def test_full_reply(self):
        valid = {"ABC1234", "DEF5678", "GHI9012"}
        text = """\
@@@ CHARACTERS
@@ name=Denji gender=male role="young man"
@@ name=Pochita gender=unknown role="devil dog"

@@@ SPEAKERS
@@ ABC1234 Denji
@@ DEF5678 Pochita
@@ GHI9012 unknown

@@@ NOISE
@@ DEF5678

@@@ STYLE
@@ register=casual
@@ mood="bleak, hungry"
@@ note="keep the rough voice"
"""
        r = _parse_reply(text, valid_keys=valid)
        assert [c.name for c in r.characters] == ["Denji", "Pochita"]
        assert r.characters[0].gender == "male"
        assert r.characters[1].role == "devil dog"
        assert r.speakers == {"ABC1234": "Denji", "DEF5678": "Pochita", "GHI9012": "unknown"}
        assert r.noise == {"DEF5678"}
        assert "register: casual" in r.style
        assert "mood: bleak, hungry" in r.style
        assert "note: keep the rough voice" in r.style

    def test_unknown_keys_dropped(self):
        text = "@@@ SPEAKERS\n@@ ZZZZ Denji\n@@ ABC1234 Denji"
        r = _parse_reply(text, valid_keys={"ABC1234"})
        assert r.speakers == {"ABC1234": "Denji"}

    def test_think_block_stripped(self):
        text = "<think>reasoning</think>\n@@@ SPEAKERS\n@@ ABC1234 Denji"
        r = _parse_reply(text, valid_keys={"ABC1234"})
        assert r.speakers == {"ABC1234": "Denji"}

    def test_malformed_lines_skipped(self):
        text = """\
@@@ SPEAKERS
@@ ABC1234 Denji
garbage line
@@ DEF5678 Pochita
@@malformed
"""
        r = _parse_reply(text, valid_keys={"ABC1234", "DEF5678"})
        assert r.speakers == {"ABC1234": "Denji", "DEF5678": "Pochita"}

    def test_empty_reply(self):
        r = _parse_reply("", valid_keys={"ABC1234"})
        assert r.speakers == {}
        assert r.characters == []
        assert r.noise == set()
        assert r.style == []


class TestNoisePages:
    def test_full_page_noise(self):
        keyed = [
            _bk("AAAA", 0, "x", page=0),
            _bk("BBBB", 1, "y", page=0),
            _bk("CCCC", 2, "z", page=1),
        ]
        # Page 0 entirely noise → marked. Page 1 not.
        pages = _full_noise_pages(keyed, {"AAAA", "BBBB"})
        assert pages == {0}

    def test_partial_page_not_noise(self):
        keyed = [
            _bk("AAAA", 0, "x", page=0),
            _bk("BBBB", 1, "y", page=0),
        ]
        pages = _full_noise_pages(keyed, {"AAAA"})
        assert pages == set()

    def test_empty(self):
        assert _full_noise_pages([], set()) == set()


class TestBriefSlice:
    def test_renders_relevant_sections(self):
        brief = ChapterBrief(
            glossary={"Denji": "Denji"},
            style_notes=["register: casual"],
            key_notes={"ABC1234": "Speaker: Denji"},
            characters=[Character(name="Denji", gender="male", role="young man")],
        )
        out = brief_slice(brief, page_indices={0}, keys=["ABC1234"])
        assert "Denji → Denji" in out
        assert "Denji (male): young man" in out
        assert "register: casual" in out
        assert "#ABC1234: Speaker: Denji" in out

    def test_empty(self):
        out = brief_slice(ChapterBrief(), page_indices=set(), keys=[])
        assert out == "(none)"

    def test_only_active_keys_get_notes(self):
        brief = ChapterBrief(key_notes={"ABC1234": "x", "DEF5678": "y"})
        out = brief_slice(brief, set(), keys=["ABC1234"])
        assert "ABC1234" in out
        assert "DEF5678" not in out
