"""tool_loop — single reusable LLM tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

from typoon.llm.ir import ContentPart, Message, Provider, ToolCallMsg, ToolResponse
from typoon.llm.tool import Tool
from typoon.runs.events import Hook, LLMCall, LLMResponse, ToolResult

_NO_HOOK = Hook()


async def tool_loop(
    provider: Provider,
    messages: list[Message],
    tools: list[Tool],
    *,
    is_done: Callable[[], bool],
    agent: str = "agent",
    max_turns: int = 20,
    hook: Hook = _NO_HOOK,
) -> int:
    """Drive a tool-calling conversation loop.

    Dispatches tool calls in parallel when the model returns multiple at once.
    Returns turns used. Raises on provider error or max_turns exceeded.
    Callers own the messages list and may pre-populate it.
    """
    tool_map = {t.definition.name: t for t in tools}
    defs = [t.definition for t in tools]

    for turn in range(1, max_turns + 1):
        hook.on(LLMCall(agent=agent, turn=turn))

        t0 = time.monotonic()
        resp = await provider.call(messages, defs)
        ms = (time.monotonic() - t0) * 1000
        hook.on(LLMResponse(agent=agent, turn=turn, tool_calls=len(resp.tool_calls), ms=ms))

        if not resp.tool_calls:
            if is_done():
                return turn
            if resp.text:
                messages.append(Message.assistant(text=resp.text))
                messages.append(Message.user_text(
                    "You must call a tool now. Do not write text — call the appropriate tool directly."
                ))
            continue

        # Sanitize: replace malformed JSON args with "{}" so the model can retry
        safe = [
            tc if _valid_json(tc.arguments)
            else ToolCallMsg(id=tc.id, name=tc.name, arguments="{}")
            for tc in resp.tool_calls
        ]
        messages.append(Message.assistant(text=resp.text, tool_calls=safe))

        results = await asyncio.gather(*[
            _call_one(tc, tool_map, agent, turn, hook)
            for tc in safe
        ])
        for tc, result in results:
            messages.append(_tool_result_msg(tc.id, result))

        if is_done():
            return turn

    raise RuntimeError(f"tool_loop: '{agent}' did not complete after {max_turns} turns")


async def _call_one(
    tc: ToolCallMsg,
    tool_map: dict[str, Tool],
    agent: str,
    turn: int,
    hook: Hook,
) -> tuple[ToolCallMsg, ToolResponse]:
    hook.on(ToolResult(agent=agent, turn=turn, tool=tc.name, result="calling…"))
    t = tool_map.get(tc.name)
    if t is None:
        result = ToolResponse(f"Unknown tool: {tc.name}")
    else:
        result = await t.call(tc)
    hook.on(ToolResult(agent=agent, turn=turn, tool=tc.name, result=result.text[:120]))
    return tc, result


def _tool_result_msg(tool_call_id: str, response: ToolResponse) -> Message:
    if response.image_data_uri:
        return Message.tool_result_parts(tool_call_id, [
            ContentPart.of_text(response.text),
            ContentPart.of_image(response.image_data_uri),
        ])
    return Message.tool_result_text(tool_call_id, response.text)


def _valid_json(s: str) -> bool:
    if not s or not s.strip():
        return True
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False
