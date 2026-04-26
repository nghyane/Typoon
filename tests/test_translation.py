"""Tests for keyed chapter brief translation pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.translation.image import encode_page_jpeg
from typoon.translation.tools.brief import submit_chapter_brief
from typoon.translation.tools.submit import SubmitArgs, submit_translations
from typoon.translation.tools.search_knowledge import SearchKnowledgeArgs
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
                {"key": key, "kind": kind, "text": text}
                for key, kind, text in items
            ],
        }),
    )])


class TestToolSchemas:
    def test_submit_translations_strict(self):
        assert submit_translations.definition.strict is True
        assert "items" in submit_translations.definition.parameters["properties"]

    def test_submit_chapter_brief_strict(self):
        assert submit_chapter_brief.definition.strict is True
        props = submit_chapter_brief.definition.parameters["properties"]
        assert "summary" in props
        assert "rules" in props
        assert "bubble_notes" in props
        assert "look_requests" not in props

    def test_submit_args_enum_kind(self):
        args = SubmitArgs.model_validate_json(json.dumps({
            "items": [
                {"key": "ABC2345", "kind": "dialogue", "text": "hello"},
                {"key": "DEF6789", "kind": "skip", "text": ""},
            ],
        }))
        assert args.items[0].kind.value == "dialogue"
        assert args.items[1].kind.value == "skip"

    def test_search_knowledge_enum_scope(self):
        args = SearchKnowledgeArgs.model_validate_json(
            json.dumps({"query": "test", "scope": "glossary"})
        )
        assert args.scope.value == "glossary"


class TestImageOverlay:
    def test_encode_with_labels(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        labels = {"ABC2345": [[10, 10], [50, 10], [50, 40], [10, 40]]}
        result = encode_page_jpeg(img, labels=labels)
        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_without_labels(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_page_jpeg(img)
        assert result.startswith("data:image/jpeg;base64,")


class TestTranslate:
    @pytest.mark.asyncio
    async def test_keyed_tool_translation(self):
        pages, session = make_session(3)
        session.context_provider = MockProvider([_brief_response()])

        async def call(messages, tools):
            keys = [b.translation_key for b in pages[0].bubbles]
            return _tool_response([(keys[0], "dialogue", "A"), (keys[1], "skip", ""), (keys[2], "dialogue", "C")])

        session.provider = MockProvider([])
        session.provider.call = call
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "", "C"]
        assert pages[0].bubbles[1].translation_status == "skip"

    @pytest.mark.asyncio
    async def test_page_agent_retry_on_missing_key(self):
        """PageAgent retries when first response misses a key."""
        pages, session = make_session(2)
        session.context_provider = MockProvider([_brief_response()])
        call_count = 0

        async def call(messages, tools):
            nonlocal call_count
            call_count += 1
            keys = [b.translation_key for b in pages[0].bubbles]
            if call_count == 1:
                return CallResponse(tool_calls=[ToolCallMsg(
                    id="b1", name="submit_chapter_brief",
                    arguments=json.dumps({
                        "summary": "s", "facts": [], "glossary": [],
                        "rules": [], "page_notes": [], "bubble_notes": [],
                    }),
                )])
            if call_count == 2:
                return _tool_response([(keys[0], "dialogue", "A")])
            return _tool_response([(keys[1], "dialogue", "B")])

        session.context_provider = MockProvider([])
        session.context_provider.call = call
        session.provider = MockProvider([])
        session.provider.call = call
        turns, err = await translate_pages(pages, session)
        assert err is None
        assert call_count >= 3
        assert [b.translated_text for b in pages[0].bubbles] == ["A", "B"]

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

    @pytest.mark.asyncio
    async def test_brief_saved_after_success(self):
        pages, session = make_session(1)
        session.context_provider = MockProvider([_brief_response()])

        async def call(messages, tools):
            key = pages[0].bubbles[0].translation_key
            return _tool_response([(key, "dialogue", "A")])

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
