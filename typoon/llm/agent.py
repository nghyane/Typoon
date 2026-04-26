"""Generic agent framework — protocol + reusable loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..app.events import Hook, LLMCall, LLMResponse, LLMText, LLMThinking, PipelineError, ToolCallStart, ToolResult
from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Provider,
    StreamEvent,
    StreamEventType,
    ToolCallMsg,
    ToolDef,
    ToolResponse,
)

SAFETY_MAX_TURNS = 50

_NO_HOOK = Hook()


@runtime_checkable
class Agent(Protocol):
    def name(self) -> str: ...
    def system_prompt(self) -> str: ...
    def user_message(self) -> Message: ...
    def tools(self) -> list[ToolDef]: ...
    async def dispatch(self, call: ToolCallMsg) -> ToolResponse: ...
    def on_text(self, text: str | None) -> None: ...
    def is_done(self) -> bool: ...
    def retry_prompt(self) -> str | None: ...
    def into_output(self) -> Any: ...


@dataclass
class RunResult:
    output: Any
    turns: int
    error: Exception | None = None


async def run(provider: Provider, agent: Agent, hook: Hook = _NO_HOOK, *, max_turns: int = SAFETY_MAX_TURNS) -> RunResult:
    """Generic agent loop. Uses stream() if available, falls back to call()."""
    messages = [Message.system(agent.system_prompt()), agent.user_message()]
    tools = agent.tools()
    name = agent.name()

    has_stream = hasattr(provider, "stream") and callable(getattr(provider, "stream", None))

    turns_used = 0
    error: Exception | None = None
    for turn in range(max_turns):
        turns_used = turn + 1
        hook.on(LLMCall(agent=name, turn=turns_used))

        t0 = time.monotonic()
        try:
            if has_stream:
                resp = await _consume_stream(provider, messages, tools, name, turns_used, hook)
            else:
                resp = await provider.call(messages, tools)
        except Exception as e:
            error = e
            hook.on(PipelineError(stage=name, error=e))
            break

        ms = (time.monotonic() - t0) * 1000
        n_tools = len(resp.tool_calls) if resp.tool_calls else 0
        hook.on(LLMResponse(agent=name, turn=turns_used, tool_calls=n_tools, ms=ms))

        if not resp.tool_calls:
            if resp.text:
                hook.on(LLMText(agent=name, turn=turns_used, delta=f"[no tool] {resp.text[:300]}"))
            agent.on_text(resp.text)
            if agent.is_done():
                break
            retry = agent.retry_prompt()
            if retry is not None:
                messages.append(Message.user_text(retry))
                continue
            break

        # Sanitize assistant tool_calls before appending to message history.
        # Streaming can produce truncated/malformed JSON args; sending them
        # back upstream triggers a 400 at the gateway before we ever see
        # the next turn. Replace bad args with "{}" and short-circuit the
        # dispatch for those calls with an error tool_result so the LLM
        # retries on the next turn.
        safe_tool_calls = []
        bad_call_ids: set[str] = set()
        for tc in resp.tool_calls:
            if _is_valid_json(tc.arguments):
                safe_tool_calls.append(tc)
            else:
                safe_tool_calls.append(ToolCallMsg(
                    id=tc.id, name=tc.name, arguments="{}",
                ))
                bad_call_ids.add(tc.id)

        messages.append(Message.assistant(text=resp.text, tool_calls=safe_tool_calls))

        for tc in resp.tool_calls:
            if tc.id in bad_call_ids:
                result = ToolResponse(
                    "Error: your previous tool call had malformed JSON arguments "
                    "and was dropped. Retry with valid JSON."
                )
            else:
                result = await agent.dispatch(tc)
            hook.on(ToolResult(agent=name, turn=turns_used, tool=tc.name, result=result.text[:120]))
            messages.append(_tool_response_to_message(tc.id, result))

        if agent.is_done():
            break

    return RunResult(output=agent.into_output(), turns=turns_used, error=error)


async def _consume_stream(
    provider: Provider,
    messages: list[Message],
    tools: list[ToolDef],
    agent_name: str,
    turn: int,
    hook: Hook,
) -> CallResponse:
    """Consume a provider stream, emitting delta events and returning a CallResponse."""
    text_parts: list[str] = []
    # index → [id, name, arg_chunks]
    pending: dict[int, list] = {}

    async for event in provider.stream(messages, tools):
        match event.type:
            case StreamEventType.THINKING_DELTA:
                hook.on(LLMThinking(agent=agent_name, turn=turn, delta=event.text))
            case StreamEventType.TEXT_DELTA:
                text_parts.append(event.text)
                hook.on(LLMText(agent=agent_name, turn=turn, delta=event.text))
            case StreamEventType.TOOL_CALL_START:
                pending[event.tool_index] = [event.tool_id, event.tool_name, []]
                hook.on(ToolCallStart(agent=agent_name, turn=turn, tool_name=event.tool_name))
            case StreamEventType.TOOL_CALL_DELTA:
                if event.tool_index in pending:
                    pending[event.tool_index][2].append(event.text)
            case StreamEventType.TOOL_CALL_DONE:
                pass  # accumulation handled via pending dict
            case StreamEventType.DONE:
                break

    tool_calls = [
        ToolCallMsg(id=entry[0], name=entry[1], arguments="".join(entry[2]))
        for _, entry in sorted(pending.items())
    ]

    text = "".join(text_parts) if text_parts else None
    return CallResponse(tool_calls=tool_calls, text=text)


def _tool_response_to_message(tool_call_id: str, response: ToolResponse) -> Message:
    if response.image_data_uri:
        return Message.tool_result_parts(tool_call_id, [
            ContentPart.of_text(response.text),
            ContentPart.of_image(response.image_data_uri),
        ])
    return Message.tool_result_text(tool_call_id, response.text)


def _is_valid_json(s: str) -> bool:
    """True if string parses as JSON. Empty string is treated as valid ({})."""
    import json
    if not s or not s.strip():
        return True
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False
