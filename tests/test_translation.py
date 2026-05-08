"""Tests for keyed chapter brief translation pipeline."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.stages.image import encode_page_jpeg
from typoon.stages.tools.brief import ChapterBriefArgs
from typoon.stages.tools.search_knowledge import SearchKnowledgeArgs
from typoon.stages.translate import translate_chapter
from typoon.stages.keys import assign_keys

from .conftest import MockProvider, make_session


def _brief_response(noise_keys: list[str] | None = None) -> CallResponse:
    args = {
        "summary": "test chapter",
        "facts": [],
        "glossary": [],
        "address": [],
        "style_notes": [],
        "page_notes": [],
        "bubble_notes": [],
    }
    calls = []
    if noise_keys:
        calls.append(ToolCallMsg(
            id="n1",
            name="mark_noise",
            arguments=json.dumps({"keys": noise_keys, "reason": "test"}),
        ))
    calls.append(ToolCallMsg(
        id="b1",
        name="submit_chapter_brief",
        arguments=json.dumps(args),
    ))
    return CallResponse(tool_calls=calls)


def _xml_response(items: list[tuple[str, str, str]]) -> CallResponse:
    lines = ["<translations>"]
    for key, kind, text in items:
        lines.append(f'<t id="{key}" kind="{kind}">{text}</t>')
    lines.append("</translations>")
    return CallResponse(text="\n".join(lines))


class TestToolSchemas:
    def test_submit_chapter_brief_schema(self):
        from typoon.llm.tool import _build_schema
        schema = _build_schema(ChapterBriefArgs)
        props = schema["properties"]
        assert "summary" in props
        assert "style_notes" in props
        assert "bubble_notes" in props
        assert "look_requests" not in props

    def test_search_knowledge_enum_scope(self):
        args = SearchKnowledgeArgs.model_validate_json(
            json.dumps({"queries": ["test"], "scope": "glossary"})
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
    async def test_keyed_xml_translation(self):
        scanned, reader, ctx = make_session(3)
        key_list = assign_keys(scanned.all_bubbles, project_id=ctx.project_id, chapter_id=ctx.chapter_id)
        keys = [bk.key for bk in key_list]
        ctx = dataclasses.replace(
            ctx, context_provider=MockProvider([_brief_response(noise_keys=[keys[1]])]),
        )

        async def call(messages, tools):
            return _xml_response([
                (keys[0], "dialogue", "A"),
                (keys[2], "dialogue", "C"),
            ])

        provider = MockProvider([])
        provider.call = call
        ctx = dataclasses.replace(ctx, translation_provider=provider)
        result, brief = await translate_chapter(scanned, reader, ctx)
        bubbles = result.all_bubbles
        assert bubbles[0].translated_text == "A"
        # Bubble 1 was flagged by the context agent as noise — translator never
        # saw it, but it must still appear in the output as kind="skip".
        assert bubbles[1].translated_text == ""
        assert bubbles[1].kind == "skip"
        assert keys[1] in brief.noise_keys
        assert bubbles[2].translated_text == "C"

    @pytest.mark.asyncio
    async def test_no_tool_call_raises(self):
        scanned, reader, ctx = make_session(1)
        ctx = dataclasses.replace(ctx, context_provider=MockProvider([_brief_response()]))

        async def call(messages, tools):
            return CallResponse(text="no tool call")

        provider = MockProvider([])
        provider.call = call
        ctx = dataclasses.replace(ctx, translation_provider=provider)
        with pytest.raises(Exception):
            await translate_chapter(scanned, reader, ctx)

    @pytest.mark.asyncio
    async def test_brief_saved_after_success(self):
        scanned, reader, ctx = make_session(1)
        ctx = dataclasses.replace(ctx, context_provider=MockProvider([_brief_response()]))
        key_list = assign_keys(scanned.all_bubbles, project_id=ctx.project_id, chapter_id=ctx.chapter_id)
        keys = [bk.key for bk in key_list]

        async def call(messages, tools):
            return _xml_response([(keys[0], "dialogue", "A")])

        provider = MockProvider([])
        provider.call = call
        ctx = dataclasses.replace(ctx, translation_provider=provider)
        _translated, brief = await translate_chapter(scanned, reader, ctx)
        await ctx.store.save_chapter_brief(ctx.chapter_id, brief.to_dict())
        saved = await ctx.store.get_recent_chapter_briefs(ctx.project_id, 99, limit=1)
        assert saved
        assert saved[0]["brief"]["summary"] == "test chapter"

    @pytest.mark.asyncio
    async def test_no_brief_tool_call_raises(self):
        scanned, reader, ctx = make_session(1)
        ctx = dataclasses.replace(ctx, context_provider=MockProvider([CallResponse(text="no tool")]))
        with pytest.raises(Exception):
            await translate_chapter(scanned, reader, ctx)
