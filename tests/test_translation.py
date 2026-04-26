"""Tests for keyed chapter brief translation."""

from __future__ import annotations

import json

import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.translation.tools.brief import submit_chapter_brief
from typoon.translation.tools.submit import SubmitArgs, submit_translations
from typoon.translation.translate import translate_pages

from .conftest import MockProvider, make_session


def _brief_response() -> CallResponse:
    return CallResponse(tool_calls=[ToolCallMsg(
        id="b1",
        name="submit_chapter_brief",
        arguments=json.dumps({
            "summary": "test chapter",
            "facts": [],
            "glossary": [],
            "rules": [],
            "page_notes": [],
            "bubble_notes": [],
        }),
    )])


def _tool_response(items: list[tuple[str, str, str]]) -> CallResponse:
    return CallResponse(tool_calls=[ToolCallMsg(
        id="c1",
        name="submit_translations",
        arguments=json.dumps({
            "items": [
                {"key": key, "status": status, "text": text}
                for key, status, text in items
            ],
        }),
    )])


class TestTool:
    def test_submit_schema_strict(self):
        assert submit_translations.definition.name == "submit_translations"
        assert submit_translations.definition.strict is True
        assert "items" in submit_translations.definition.parameters["properties"]

    def test_brief_schema_strict(self):
        assert submit_chapter_brief.definition.name == "submit_chapter_brief"
        assert submit_chapter_brief.definition.strict is True
        props = submit_chapter_brief.definition.parameters["properties"]
        assert "summary" in props
        assert "rules" in props
        assert "bubble_notes" in props
        assert "look_requests" not in props

    def test_submit_args_parse(self):
        args = SubmitArgs.model_validate_json(json.dumps({
            "items": [
                {"key": "ABC2345", "status": "ok", "text": "hello"},
                {"key": "DEF6789", "status": "skip", "text": ""},
            ],
        }))
        assert args.items[0].key == "ABC2345"
        assert args.items[1].status.value == "skip"


class TestTranslate:
    @pytest.mark.asyncio
    async def test_keyed_tool_translation(self):
        pages, session = make_session(3)
        session.context_provider = MockProvider([_brief_response()])

        async def call(messages, tools):
            keys = [b.translation_key for b in pages[0].bubbles]
            return _tool_response([(keys[0], "ok", "A"), (keys[1], "skip", ""), (keys[2], "ok", "C")])

        session.provider = MockProvider([])
        session.provider.call = call
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert turns == 2
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "", "C"]
        assert pages[0].bubbles[1].translation_status == "skip"

    @pytest.mark.asyncio
    async def test_no_tool_call_raises(self):
        pages, session = make_session(1)
        session.context_provider = MockProvider([_brief_response()])

        async def call(messages, tools):
            return CallResponse(text="no tool call")

        session.provider = MockProvider([])
        session.provider.call = call
        turns, err = await translate_pages(pages, session)
        assert err is not None
        assert "submit_translations" in str(err)

    @pytest.mark.asyncio
    async def test_brief_saved_after_success(self):
        pages, session = make_session(1)
        session.context_provider = MockProvider([_brief_response()])

        async def call(messages, tools):
            key = pages[0].bubbles[0].translation_key
            return _tool_response([(key, "ok", "A")])

        session.provider = MockProvider([])
        session.provider.call = call
        turns, err = await translate_pages(pages, session)
        assert err is None
        saved = await session.store.get_recent_chapter_briefs(session.project_id, 99, limit=1)
        assert saved
        assert saved[0]["brief"]["summary"] == "test chapter"

    @pytest.mark.asyncio
    async def test_no_brief_tool_call_raises(self):
        pages, session = make_session(1)
        session.context_provider = MockProvider([CallResponse(text="no tool")])
        session.provider = MockProvider([])
        turns, err = await translate_pages(pages, session)
        assert err is not None
