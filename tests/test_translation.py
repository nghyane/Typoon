"""Tests for the two-pass translation module."""

from __future__ import annotations

import json

import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.translation.prompt import SYSTEM, load_policy
from typoon.translation.tools.submit import SubmitArgs, submit_translations
from typoon.translation.translate import translate_pages

from .conftest import MockProvider, make_session, make_translate_response


class TestTool:
    def test_submit_schema_strict(self):
        assert submit_translations.definition.name == "submit_translations"
        assert submit_translations.definition.strict is True
        assert "edits" in submit_translations.definition.parameters["properties"]

    def test_submit_args_parse(self):
        args = SubmitArgs.model_validate_json(json.dumps({
            "edits": [
                {"id": "p0_b0", "text": "hello", "unclear": False},
                {"id": "p0_b1", "text": "", "unclear": True},
            ],
        }))
        assert args.edits[0].id == "p0_b0"
        assert args.edits[1].unclear is True


class TestPrompts:
    def test_system_contains_lang_pair(self):
        result = SYSTEM.format(
            source_lang="en", target_lang="vi",
            source_policy=load_policy("source_en.md"),
            target_policy=load_policy("target_vi.md"),
        )
        assert "en → vi" in result
        assert "submit_translations" in result

    def test_load_policy_files(self):
        assert "English" in load_policy("source_en.md")
        assert "Vietnamese" in load_policy("target_vi.md")


class TestTranslatePasses:
    @pytest.mark.asyncio
    async def test_pass1_only_when_all_clear(self):
        """Pass 2 skipped when no bubbles marked unclear."""
        pages, session = make_session(3)
        session.provider = MockProvider([
            make_translate_response([
                ("p0_b0", "A"), ("p0_b1", "B"), ("p0_b2", "C"),
            ]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert turns == 1
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_pass2_resolves_unclear(self):
        """Pass 2 runs for unclear bubbles and overwrites their text."""
        pages, session = make_session(3)
        session.provider = MockProvider([
            # Pass 1: one bubble unclear
            make_translate_response(
                [("p0_b0", "A"), ("p0_b1", "placeholder"), ("p0_b2", "C")],
                unclear=["p0_b1"],
            ),
            # Pass 2: resolve unclear
            make_translate_response([("p0_b1", "B")]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert turns == 2
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_multiple_submit_calls_per_turn(self):
        """LLM may batch via multiple submit_translations calls; all merge."""
        pages, session = make_session(4)
        session.provider = MockProvider([
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="submit_translations",
                            arguments=json.dumps({"edits": [
                                {"id": "p0_b0", "text": "A", "unclear": False},
                                {"id": "p0_b1", "text": "B", "unclear": False},
                            ]})),
                ToolCallMsg(id="c2", name="submit_translations",
                            arguments=json.dumps({"edits": [
                                {"id": "p0_b2", "text": "C", "unclear": False},
                                {"id": "p0_b3", "text": "D", "unclear": False},
                            ]})),
            ]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert turns == 1
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B", "C", "D"]

    @pytest.mark.asyncio
    async def test_pass3_fills_missing(self):
        """Pass 3 triggered when LLM silently skipped some IDs."""
        pages, session = make_session(3)
        session.provider = MockProvider([
            # Pass 1: only two of three
            make_translate_response([("p0_b0", "A"), ("p0_b2", "C")]),
            # Pass 3: fill the missing one (no pass 2 needed — no unclear)
            make_translate_response([("p0_b1", "B")]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert turns == 2
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_corrupt_tool_args_tolerated(self):
        """Malformed tool args don't crash; missing-ids retry fills the gap."""
        pages, session = make_session(2)
        session.provider = MockProvider([
            # Pass 1: one good, one corrupt
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="submit_translations",
                            arguments=json.dumps({"edits": [
                                {"id": "p0_b0", "text": "A", "unclear": False},
                            ]})),
                ToolCallMsg(id="c2", name="submit_translations",
                            arguments='{"edits": [{id": broken'),  # corrupt
            ]),
            # Pass 3: fill the missing
            make_translate_response([("p0_b1", "B")]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_glossary_in_prompt(self):
        pages, session = make_session(1, glossary={"Chairman": "Hội trưởng"})
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "Hội trưởng nói")]),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == "Hội trưởng nói"

    @pytest.mark.asyncio
    async def test_empty_text_for_noise(self):
        pages, session = make_session(1)
        pages[0].bubbles[0].source_text = "www.example.com"
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "")]),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == ""
