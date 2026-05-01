"""OpenAI provider — native SDK adapter.

Serializes IR → OpenAI chat format, calls via `openai` SDK,
parses response back to CallResponse IR. Supports streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai

from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCallMsg,
    ToolDef,
)

_TIMEOUT = 180


class OpenAIProvider:
    """OpenAI provider — supports standard OpenAI, Cloudflare Gateway, and proxies."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        reasoning_effort: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        kwargs: dict = {"api_key": api_key, "base_url": base_url, "timeout": _TIMEOUT}
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        kwargs: dict = {
            "messages": [_serialize_message(m) for m in messages],
        }
        # model="" lets gateway auto-route (Cloudflare, etc.)
        if self._model:
            kwargs["model"] = self._model
        if tools:
            kwargs["tools"] = [_serialize_tool(t) for t in tools]
            kwargs["parallel_tool_calls"] = True
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        resp = await self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise RuntimeError(
                f"OpenAI returned empty response (no choices). "
                f"Check endpoint/model config. model={self._model}"
            )
        choice = resp.choices[0]

        tool_calls: list[ToolCallMsg] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCallMsg(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        return CallResponse(tool_calls=tool_calls, text=choice.message.content)

    async def stream(self, messages: list[Message], tools: list[ToolDef]) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "messages": [_serialize_message(m) for m in messages],
            "stream": True,
        }
        if self._model:
            kwargs["model"] = self._model
        if tools:
            kwargs["tools"] = [_serialize_tool(t) for t in tools]
            kwargs["parallel_tool_calls"] = True
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = await self._client.chat.completions.create(**kwargs)

        pending: dict[int, tuple[str, str]] = {}  # index → (id, name)
        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta

            # Thinking tokens (o-series models)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield StreamEvent(type=StreamEventType.THINKING_DELTA, text=delta.reasoning_content)

            # Text content
            if delta.content:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.content)

            # Tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending and tc_delta.function:
                        pending[idx] = (tc_delta.id or "", tc_delta.function.name or "")
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_index=idx,
                            tool_id=tc_delta.id or "",
                            tool_name=tc_delta.function.name or "",
                        )
                    if tc_delta.function and tc_delta.function.arguments:
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_index=idx,
                            text=tc_delta.function.arguments,
                        )

            if choice.finish_reason is not None:
                for idx, (tid, tname) in sorted(pending.items()):
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DONE,
                        tool_index=idx,
                        tool_id=tid,
                        tool_name=tname,
                    )
                yield StreamEvent(type=StreamEventType.DONE)


# ── IR → OpenAI serialization ────────────────────────────────────────


def _serialize_message(msg: Message) -> dict:
    match msg.role:
        case Role.SYSTEM:
            return {"role": "system", "content": msg.text or ""}
        case Role.USER:
            return {"role": "user", "content": _serialize_parts(msg.parts)}
        case Role.ASSISTANT:
            m: dict = {"role": "assistant"}
            if msg.text:
                m["content"] = msg.text
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            return m
        case Role.TOOL_RESULT:
            content = _serialize_parts(msg.parts)
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            }


def _serialize_parts(parts: list[ContentPart]) -> str | list[dict]:
    if len(parts) == 1 and parts[0].text is not None and parts[0].image_data_uri is None:
        return parts[0].text
    result = []
    for p in parts:
        if p.text is not None:
            result.append({"type": "text", "text": p.text})
        if p.image_data_uri is not None:
            result.append({
                "type": "image_url",
                "image_url": {"url": p.image_data_uri, "detail": "low"},
            })
    return result


def _serialize_tool(tool: ToolDef) -> dict:
    return {"type": "function", "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }}
