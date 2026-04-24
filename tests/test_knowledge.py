"""Tests for the knowledge consolidation agent."""

from __future__ import annotations

import json
import pytest

from typoon.app.events import Hook
from typoon.domain.bubble import Session
from typoon.llm.ir import CallResponse, ToolCallMsg
from typoon.translation.knowledge import consolidate

from .conftest import MockProvider, MockSource, MockStore


def _session(store: MockStore, responses: list[CallResponse]) -> Session:
    p = MockProvider(responses)
    return Session(
        store=store, source=MockSource(), project_id=1,
        source_lang="en", target_lang="vi",
        provider=p, context_provider=p, hook=Hook(),
    )


@pytest.mark.asyncio
async def test_updates_snapshot():
    """Snapshot update — covers store.save_knowledge + get_knowledge chain."""
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
async def test_all_tools_one_call():
    """All three knowledge tools dispatched in one LLM turn."""
    store = MockStore()
    session = _session(store, [
        CallResponse(tool_calls=[
            ToolCallMsg(id="c1", name="update_snapshot",
                        arguments=json.dumps({"snapshot": "Updated"})),
            ToolCallMsg(id="c2", name="add_note",
                        arguments=json.dumps({"note_type": "event", "content": "Fight scene"})),
            ToolCallMsg(id="c3", name="upsert_glossary",
                        arguments=json.dumps({"source_term": "Dark Guild",
                                              "target_term": "Hội Bóng Tối", "notes": ""})),
        ]),
    ])
    result = await consolidate(session, chapter=3, pairs=[("a", "b")])
    assert result.error is None
    assert store.snapshots[3] == "Updated"
    assert store.notes[0]["type"] == "event"
    assert store.glossary["Dark Guild"] == "Hội Bóng Tối"
