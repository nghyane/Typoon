"""Tests for the knowledge consolidation agent."""

from __future__ import annotations

import json
import pytest

from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.events import Hook
from typoon.domain.bubble import Session
from typoon.translation.knowledge import consolidate

from .conftest import MockProvider, MockStore, MockSource


def _session(store: MockStore, responses: list[CallResponse]) -> Session:
    p = MockProvider(responses)
    return Session(
        store=store, source=MockSource(), project_id=1,
        source_lang="en", target_lang="vi",
        provider=p, context_provider=p, hook=Hook(),
    )


class TestKnowledgeAgent:
    @pytest.mark.asyncio
    async def test_updates_snapshot(self):
        store = MockStore()
        session = _session(store, [
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="update_snapshot",
                            arguments=json.dumps({"snapshot": "Min-jun: protagonist"})),
            ]),
        ])
        result = await consolidate(session, chapter=1, pairs=[("hello", "xin chào")])
        assert result.error is None
        assert store.snapshots[1] == "Min-jun: protagonist"

    @pytest.mark.asyncio
    async def test_adds_notes(self):
        store = MockStore()
        session = _session(store, [
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="add_note",
                            arguments=json.dumps({"note_type": "character", "content": "Seo-yeon introduced"})),
            ]),
        ])
        await consolidate(session, chapter=2, pairs=[("x", "y")])
        assert len(store.notes) == 1
        assert store.notes[0]["type"] == "character"

    @pytest.mark.asyncio
    async def test_upserts_glossary(self):
        store = MockStore()
        session = _session(store, [
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="upsert_glossary",
                            arguments=json.dumps({"source_term": "Chairman", "target_term": "Hội trưởng", "notes": ""})),
            ]),
        ])
        await consolidate(session, chapter=1, pairs=[("Chairman!", "Hội trưởng!")])
        assert store.glossary["Chairman"] == "Hội trưởng"

    @pytest.mark.asyncio
    async def test_all_tools_one_call(self):
        store = MockStore()
        session = _session(store, [
            CallResponse(tool_calls=[
                ToolCallMsg(id="c1", name="update_snapshot", arguments=json.dumps({"snapshot": "Updated"})),
                ToolCallMsg(id="c2", name="add_note", arguments=json.dumps({"note_type": "event", "content": "Fight scene"})),
                ToolCallMsg(id="c3", name="upsert_glossary", arguments=json.dumps({"source_term": "Dark Guild", "target_term": "Hội Bóng Tối", "notes": ""})),
            ]),
        ])
        result = await consolidate(session, chapter=3, pairs=[("a", "b")])
        assert result.error is None
        assert store.snapshots[3] == "Updated"
        assert len(store.notes) == 1
        assert store.glossary["Dark Guild"] == "Hội Bóng Tối"
