"""Tests for the translation agent."""

from __future__ import annotations

import json

import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.translation.agent import translate_pages
from typoon.translation.prompt import SYSTEM, load_policy
from typoon.translation.tools import build_tools

from .conftest import MockProvider, make_session, make_translate_response


class TestTools:
    def test_translate_schema_strict(self):
        from typoon.translation.tools.translate import translate
        assert translate.definition.name == "translate"
        assert translate.definition.strict is True

    def test_minimal_tools(self):
        tools = build_tools(has_images=False, has_glossary=False, has_context=False)
        assert len(tools) == 1

    def test_all_tools(self):
        tools = build_tools(has_images=True, has_glossary=True, has_context=True)
        assert {t.name for t in tools} == {"translate", "view_page", "view_bubble", "search_glossary", "get_context"}


class TestPrompts:
    def test_system_contains_lang_pair(self):
        result = SYSTEM.format(source_lang="en", target_lang="vi",
                               source_policy=load_policy("source_en.md"),
                               target_policy=load_policy("target_vi.md"))
        assert "en → vi" in result
        assert "ONLY with tool calls" in result

    def test_load_policy_files(self):
        assert "English" in load_policy("source_en.md")
        assert "Vietnamese" in load_policy("target_vi.md")


class TestTranslateAgent:
    @pytest.mark.asyncio
    async def test_all_in_one_turn(self):
        pages, session = make_session(3)
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "A"), ("p0_b1", "B"), ("p0_b2", "C")]),
        ])
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert pages[0].bubbles[0].translated_text == "A"
        assert pages[0].bubbles[2].translated_text == "C"

    @pytest.mark.asyncio
    async def test_multi_turn(self):
        pages, session = make_session(3)
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "A")]),
            make_translate_response([("p0_b1", "B"), ("p0_b2", "C")]),
        ])
        await translate_pages(pages, session)
        assert all(b.translated_text for b in pages[0].bubbles)

    @pytest.mark.asyncio
    async def test_partial_preserves_results(self):
        pages, session = make_session(2)
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "A")]),
            CallResponse(text="done"),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == "A"
        assert pages[0].bubbles[1].translated_text is None

    @pytest.mark.asyncio
    async def test_dedup_keeps_latest(self):
        pages, session = make_session(1)
        session.provider = MockProvider([
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="translate", arguments=json.dumps({"translations": [{"id": "p0_b0", "translated_text": "first"}]})),
                ToolCallMsg(id="c2", name="translate", arguments=json.dumps({"translations": [{"id": "p0_b0", "translated_text": "revised"}]})),
            ]),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == "revised"

    @pytest.mark.asyncio
    async def test_glossary_injection(self):
        pages, session = make_session(1, glossary={"Chairman": "Hội trưởng"})
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "Hội trưởng nói")]),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == "Hội trưởng nói"

    @pytest.mark.asyncio
    async def test_noise_empty_string(self):
        pages, session = make_session(1)
        pages[0].bubbles[0].source_text = "www.example.com"
        session.provider = MockProvider([
            make_translate_response([("p0_b0", "")]),
        ])
        await translate_pages(pages, session)
        assert pages[0].bubbles[0].translated_text == ""
