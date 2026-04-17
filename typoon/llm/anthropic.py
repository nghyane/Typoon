"""Anthropic provider — native SDK adapter.

Serializes IR → Anthropic messages format, calls via `anthropic` SDK,
parses response back to CallResponse IR. Supports prompt caching + streaming.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anthropic

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

_MAX_TOKENS = 64_000


class AnthropicProvider:
    """Anthropic native provider with prompt caching + streaming."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = model
        self._max_tokens = max_tokens

    def _build_kwargs(self, messages: list[Message], tools: list[ToolDef]) -> dict:
        system = _extract_system(messages)
        api_messages = _serialize_messages(messages)
        api_tools = _serialize_tools(tools)

        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system is not None:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools
        return kwargs

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        tool_calls: list[ToolCallMsg] = []
        text_parts: list[str] = []

        async for event in self.stream(messages, tools):
            match event.type:
                case StreamEventType.TEXT_DELTA:
                    text_parts.append(event.text)
                case StreamEventType.TOOL_CALL_DONE:
                    tool_calls.append(ToolCallMsg(
                        id=event.tool_id, name=event.tool_name, arguments=event.text,
                    ))

        text = "".join(text_parts) if text_parts else None
        return CallResponse(tool_calls=tool_calls, text=text)

    async def stream(self, messages: list[Message], tools: list[ToolDef]) -> AsyncIterator[StreamEvent]:
        kwargs = self._build_kwargs(messages, tools)

        pending_tools: dict[int, dict] = {}

        async with self._client.messages.stream(**kwargs) as raw:
            async for event in raw:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        pending_tools[event.index] = {
                            "id": block.id, "name": block.name, "input_json": "",
                        }
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_index=event.index,
                            tool_id=block.id,
                            tool_name=block.name,
                        )
                    elif block.type == "thinking":
                        pass  # deltas arrive via thinking_delta
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.text)
                    elif delta.type == "input_json_delta":
                        if event.index in pending_tools:
                            pending_tools[event.index]["input_json"] += delta.partial_json
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_index=event.index,
                                text=delta.partial_json,
                            )
                    elif delta.type == "thinking_delta":
                        yield StreamEvent(type=StreamEventType.THINKING_DELTA, text=delta.thinking)
                elif event.type == "content_block_stop":
                    if event.index in pending_tools:
                        t = pending_tools.pop(event.index)
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DONE,
                            tool_index=event.index,
                            tool_id=t["id"],
                            tool_name=t["name"],
                            text=t["input_json"],
                        )

        yield StreamEvent(type=StreamEventType.DONE)


# ── IR → Anthropic serialization ─────────────────────────────────────


def _extract_system(messages: list[Message]) -> list[dict] | None:
    """Extract system message with cache_control for prompt caching."""
    for msg in messages:
        if msg.role == Role.SYSTEM and msg.text:
            return [{"type": "text", "text": msg.text, "cache_control": {"type": "ephemeral"}}]
    return None


def _serialize_messages(messages: list[Message]) -> list[dict]:
    result = []
    for msg in messages:
        match msg.role:
            case Role.SYSTEM:
                continue  # handled separately
            case Role.USER:
                result.append({"role": "user", "content": _serialize_parts(msg.parts)})
            case Role.ASSISTANT:
                blocks: list[dict] = []
                if msg.text:
                    blocks.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    inp = json.loads(tc.arguments) if tc.arguments else {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": inp,
                    })
                result.append({"role": "assistant", "content": blocks})
            case Role.TOOL_RESULT:
                text = "\n".join(p.text for p in msg.parts if p.text)
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": text,
                    }],
                })
    return result


def _serialize_parts(parts: list[ContentPart]) -> str | list[dict]:
    if len(parts) == 1 and parts[0].text is not None and parts[0].image_data_uri is None:
        return parts[0].text
    blocks: list[dict] = []
    for p in parts:
        if p.text is not None:
            blocks.append({"type": "text", "text": p.text})
        if p.image_data_uri is not None:
            data_uri = p.image_data_uri
            if data_uri.startswith("data:image/jpeg;base64,"):
                b64 = data_uri[len("data:image/jpeg;base64,"):]
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
            elif data_uri.startswith("data:image/png;base64,"):
                b64 = data_uri[len("data:image/png;base64,"):]
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
            else:
                blocks.append({"type": "text", "text": "[unsupported image format]"})
    return blocks


def _serialize_tools(tools: list[ToolDef]) -> list[dict]:
    n = len(tools)
    result = []
    for i, t in enumerate(tools):
        tool: dict = {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        # Cache breakpoint on last tool — caches system + all tools prefix
        if i == n - 1:
            tool["cache_control"] = {"type": "ephemeral"}
        result.append(tool)
    return result
