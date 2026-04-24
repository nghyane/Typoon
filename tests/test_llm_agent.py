"""Tests for the generic agent loop."""

import pytest

from typoon.llm.ir import CallResponse, Message, ToolCallMsg, ToolDef, ToolResponse
from typoon.llm.agent import run

from .conftest import MockProvider


class CollectAgent:
    """Collects tool call arguments until target count reached."""

    def __init__(self, target: int) -> None:
        self.items: list[str] = []
        self._target = target

    def name(self) -> str: return "test"
    def system_prompt(self) -> str: return "test"
    def user_message(self) -> Message: return Message.user_text("go")
    def tools(self) -> list[ToolDef]: return [ToolDef("add", "add item", {"type": "object"})]
    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        if call.name == "add":
            self.items.append(call.arguments)
        return ToolResponse("ok")
    def on_text(self, text: str | None) -> None: pass
    def is_done(self) -> bool: return len(self.items) >= self._target
    def retry_prompt(self) -> str | None: return None
    def into_output(self) -> list[str]: return list(self.items)


class RetryAgent:
    """Agent that retries once when no tool calls received."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self._retries = 0

    def name(self) -> str: return "retry"
    def system_prompt(self) -> str: return "test"
    def user_message(self) -> Message: return Message.user_text("go")
    def tools(self) -> list[ToolDef]: return [ToolDef("add", "add", {"type": "object"})]
    async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
        self.items.append(call.arguments)
        return ToolResponse("ok")
    def on_text(self, text: str | None) -> None: pass
    def is_done(self) -> bool: return len(self.items) >= 2
    def retry_prompt(self) -> str | None:
        if self._retries >= 1: return None
        self._retries += 1
        return "please continue"
    def into_output(self) -> list[str]: return list(self.items)


class TextAgent:
    """Agent that captures text response."""

    def __init__(self) -> None:
        self.answer: str | None = None

    def name(self) -> str: return "text"
    def system_prompt(self) -> str: return "answer"
    def user_message(self) -> Message: return Message.user_text("question")
    def tools(self) -> list[ToolDef]: return [ToolDef("search", "search", {"type": "object"})]
    async def dispatch(self, call: ToolCallMsg) -> ToolResponse: return ToolResponse("search results")
    def on_text(self, text: str | None) -> None: self.answer = text
    def is_done(self) -> bool: return self.answer is not None
    def retry_prompt(self) -> str | None: return None
    def into_output(self) -> str: return self.answer or ""


@pytest.mark.asyncio
async def test_multi_turn():
    provider = MockProvider([
        CallResponse(tool_calls=[
            ToolCallMsg(id="c1", name="add", arguments='{"v":"item1"}'),
            ToolCallMsg(id="c2", name="add", arguments='{"v":"item2"}'),
        ]),
        CallResponse(tool_calls=[
            ToolCallMsg(id="c3", name="add", arguments='{"v":"item3"}'),
        ]),
    ])
    result = await run(provider, CollectAgent(target=3))
    assert result.output == ['{"v":"item1"}', '{"v":"item2"}', '{"v":"item3"}']
    assert result.turns == 2


@pytest.mark.asyncio
async def test_early_stop_when_done():
    """Agent stops as soon as is_done() returns True, regardless of remaining turns."""
    provider = MockProvider([
        CallResponse(tool_calls=[ToolCallMsg(id="c1", name="add", arguments='{"v":"only_one"}')]),
    ])
    result = await run(provider, CollectAgent(target=1))
    assert result.output == ['{"v":"only_one"}']
    assert result.turns == 1


@pytest.mark.asyncio
async def test_stops_on_empty_response():
    """If model returns no tool calls and agent doesn't retry, stop with partial output."""
    provider = MockProvider([CallResponse(text="done")])
    result = await run(provider, CollectAgent(target=99))
    assert result.output == []
    assert result.turns == 1


@pytest.mark.asyncio
async def test_text_response():
    provider = MockProvider([
        CallResponse(tool_calls=[ToolCallMsg(id="c1", name="search", arguments="{}")]),
        CallResponse(text="The answer is 42"),
    ])
    result = await run(provider, TextAgent())
    assert result.output == "The answer is 42"
    assert result.turns == 2


@pytest.mark.asyncio
async def test_malformed_json_args_dont_reach_dispatch():
    """Truncated/corrupt tool args are short-circuited with an error
    tool_result so the LLM can retry, instead of being forwarded to the
    provider where they would trigger a 400."""
    dispatched: list[str] = []

    class GuardAgent:
        def __init__(self) -> None:
            self.items: list[str] = []
            self._done = False

        def name(self) -> str: return "guard"
        def system_prompt(self) -> str: return "test"
        def user_message(self) -> Message: return Message.user_text("go")
        def tools(self) -> list[ToolDef]: return [ToolDef("add", "add", {"type": "object"})]
        async def dispatch(self, call: ToolCallMsg) -> ToolResponse:
            dispatched.append(call.arguments)
            self.items.append(call.arguments)
            return ToolResponse("ok")
        def on_text(self, text: str | None) -> None: pass
        def is_done(self) -> bool: return self._done
        def retry_prompt(self) -> str | None:
            self._done = True  # stop after one retry turn
            return None
        def into_output(self) -> list[str]: return list(self.items)

    provider = MockProvider([
        # First turn: corrupt JSON — should NOT be dispatched
        CallResponse(tool_calls=[ToolCallMsg(id="c1", name="add", arguments='{id": "broken')]),
        # Second turn: clean JSON — should be dispatched
        CallResponse(tool_calls=[ToolCallMsg(id="c2", name="add", arguments='{"ok": true}')]),
    ])
    await run(provider, GuardAgent())
    assert dispatched == ['{"ok": true}']


@pytest.mark.asyncio
async def test_retry():
    provider = MockProvider([
        CallResponse(tool_calls=[ToolCallMsg(id="c1", name="add", arguments='{"v":"first"}')]),
        CallResponse(text="thinking..."),
        CallResponse(tool_calls=[ToolCallMsg(id="c2", name="add", arguments='{"v":"second"}')]),
    ])
    result = await run(provider, RetryAgent())
    assert result.output == ['{"v":"first"}', '{"v":"second"}']
