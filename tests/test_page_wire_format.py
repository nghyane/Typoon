"""Tests for the line-sentinel translation wire format (typoon.stages.page)."""

from __future__ import annotations

import pytest

from typoon.domain.scan import Box, Bubble, BubbleKey
from typoon.stages.page import (
    TranslationOp,
    _build_window_prompt,
    _parse_translation_reply,
)


_BOX = Box(
    polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
    fit=[0, 0, 10, 5],
    erase=[0, 0, 10, 5],
    text=[0, 0, 10, 5],
)


def _bk(key: str, idx: int, text: str, page: int = 0) -> BubbleKey:
    return BubbleKey(
        key=key,
        bubble=Bubble(
            idx=idx, page_index=page, source_text=text,
            confidence=0.9, box=_BOX,
        ),
    )


@pytest.fixture
def key_map() -> dict[str, BubbleKey]:
    return {
        "ABC1234": _bk("ABC1234", 0, "hello"),
        "DEF5678": _bk("DEF5678", 1, "BOOM"),
        "GHJ9KLM": _bk("GHJ9KLM", 2, "context only text"),
    }


class TestParseReply:
    def test_happy_path(self, key_map):
        text = (
            "preamble junk\n"
            "@@ ABC1234 dialogue\n"
            "xin chào\n"
            "@@ DEF5678 sfx\n"
            "RẦM\n"
        )
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert ops == [
            TranslationOp(key="ABC1234", kind="dialogue", text="xin chào"),
            TranslationOp(key="DEF5678", kind="sfx", text="RẦM"),
        ]

    def test_code_fence_wrap_stripped(self, key_map):
        text = "```\n@@ ABC1234 dialogue\nhi\n```"
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops == [TranslationOp(key="ABC1234", kind="dialogue", text="hi")]

    def test_think_block_stripped(self, key_map):
        text = "<think>reasoning</think>\n@@ ABC1234 dialogue\nhi"
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops == [TranslationOp(key="ABC1234", kind="dialogue", text="hi")]

    def test_multiline_body_preserved(self, key_map):
        text = "@@ ABC1234 dialogue\nline 1\nline 2\n\nline 4\n"
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops[0].text == "line 1\nline 2\n\nline 4"

    def test_duplicate_key_keeps_first(self, key_map):
        text = (
            "@@ ABC1234 dialogue\nfirst\n"
            "@@ ABC1234 dialogue\nsecond\n"
        )
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops == [TranslationOp(key="ABC1234", kind="dialogue", text="first")]

    def test_unknown_key_dropped(self, key_map):
        ops = _parse_translation_reply("@@ ZZZZZZZ dialogue\nfoo", {"ZZZZZZZ"}, key_map)
        assert ops == []

    def test_inactive_key_dropped(self, key_map):
        # ABC1234 in key_map but not in active set (context only)
        ops = _parse_translation_reply(
            "@@ ABC1234 dialogue\nfoo\n@@ DEF5678 sfx\nbar",
            {"DEF5678"}, key_map,
        )
        assert [o.key for o in ops] == ["DEF5678"]

    def test_invalid_kind_drops_block(self, key_map):
        # Header regex demands dialogue|sfx — narration is not a valid header
        # at all, so the entire block belongs to the *previous* block's body
        # (here: none). Either way, ABC1234 produces no op.
        text = (
            "@@ ABC1234 narration\n"
            "ignored\n"
            "@@ DEF5678 dialogue\n"
            "ok\n"
        )
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert [o.key for o in ops] == ["DEF5678"]

    def test_auto_skip_rubble_forced(self):
        km = {"AAA0001": _bk("AAA0001", 0, "1")}  # single digit → auto skip
        ops = _parse_translation_reply("@@ AAA0001 dialogue\nbogus", {"AAA0001"}, km)
        assert ops == [TranslationOp(key="AAA0001", kind="skip", text="")]

    def test_empty_body_dropped(self, key_map):
        text = "@@ ABC1234 dialogue\n@@ DEF5678 sfx\nRẦM"
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert [o.key for o in ops] == ["DEF5678"]

    def test_header_must_be_at_column_zero(self, key_map):
        # Leading whitespace disqualifies a header — body absorbs it.
        text = "@@ ABC1234 dialogue\nhi\n   @@ DEF5678 sfx\nignored"
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        # ABC1234 body includes the indented "@@ DEF5678 sfx" line and "ignored".
        assert ops[0].text == "hi\n   @@ DEF5678 sfx\nignored"
        assert len(ops) == 1

    def test_model_skip_for_chrome(self, key_map):
        # Model can declare a leaked-chrome bubble via `kind=skip`. Body
        # is ignored — empty or anything else, both produce a skip op.
        text = "@@ ABC1234 skip\n@@ DEF5678 dialogue\nbar"
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert [(o.key, o.kind, o.text) for o in ops] == [
            ("ABC1234", "skip", ""),
            ("DEF5678", "dialogue", "bar"),
        ]

    def test_model_skip_case_insensitive(self, key_map):
        # The kind regex is IGNORECASE; downstream sees lowercase.
        ops = _parse_translation_reply(
            "@@ ABC1234 SKIP", {"ABC1234"}, key_map,
        )
        assert ops == [TranslationOp(key="ABC1234", kind="skip", text="")]

    def test_model_skip_with_body_ignores_body(self, key_map):
        # If the model emits `kind=skip` with a body anyway (against the
        # prompt rule), we still treat it as skip — the body is dropped.
        text = "@@ ABC1234 skip\nstray text the model wrote"
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops == [TranslationOp(key="ABC1234", kind="skip", text="")]


class TestBuildPrompt:
    def test_active_flag_and_context(self, key_map):
        user = _build_window_prompt(
            "CONTEXT BLOCK",
            ["ABC1234", "DEF5678"],
            {"ABC1234"},
            key_map,
        )
        assert "CONTEXT BLOCK" in user
        # Input sentinel is `>>>`, distinct from output `@@` so the model
        # cannot mirror an input header back as a fake output header.
        assert ">>> ABC1234 page=0 active\nhello" in user
        assert ">>> DEF5678 page=0\nBOOM" in user
        assert "@@" not in user  # never use output sentinel on the way in


class TestParseTolerance:
    """Cases the parser is intentionally tolerant about."""

    def test_kind_case_normalized_to_lowercase(self, key_map):
        # Small models often emit `Dialogue` / `SFX` against instructions.
        # IGNORECASE on the regex + .lower() on the captured kind keeps
        # the block instead of silently dropping it.
        text = "@@ ABC1234 Dialogue\nhello\n@@ DEF5678 SFX\nBOOM"
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert [(o.key, o.kind) for o in ops] == [
            ("ABC1234", "dialogue"),
            ("DEF5678", "sfx"),
        ]

    def test_input_sentinel_in_output_is_rejected(self, key_map):
        # If the model mirrors `>>>` (input) instead of `@@` (output), the
        # block must NOT be parsed — that’s exactly the failure mode the
        # asymmetric sentinels are designed to surface.
        text = ">>> ABC1234 dialogue\nshould not be picked up"
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert ops == []

    def test_key_must_be_exactly_seven_chars(self, key_map):
        # assign_keys always emits 7-char keys. A 6- or 8-char token in the
        # header is malformed and must be rejected so it doesn’t shadow a
        # valid block later in the stream.
        text = (
            "@@ SHORT1 dialogue\n"
            "ignored\n"
            "@@ TOOLONG8 dialogue\n"
            "ignored\n"
            "@@ ABC1234 dialogue\n"
            "ok\n"
        )
        ops = _parse_translation_reply(text, {"ABC1234"}, key_map)
        assert [o.key for o in ops] == ["ABC1234"]

    def test_trailing_garbage_after_kind_rejected(self, key_map):
        # We deliberately stay strict on `\s*$` after kind. A model that
        # adds `page=N` or comments must learn via retry, not silent accept.
        text = (
            "@@ ABC1234 dialogue page=3\n"
            "ignored body\n"
            "@@ DEF5678 sfx\n"
            "RẦM\n"
        )
        ops = _parse_translation_reply(text, {"ABC1234", "DEF5678"}, key_map)
        assert [o.key for o in ops] == ["DEF5678"]
