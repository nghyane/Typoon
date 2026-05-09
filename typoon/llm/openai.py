"""OpenAI provider — native SDK adapter.

Serializes IR → OpenAI chat format, calls via `openai` SDK,
parses response back to CallResponse IR. Supports streaming.
"""

from __future__ import annotations

import openai

from ._retry import parse_retry_after_header, with_retry
from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Role,
    ToolCallMsg,
    ToolDef,
)

_TIMEOUT = 300


def _parse_retry_after(exc: BaseException) -> float | None:
    """Extract Retry-After from an OpenAI SDK exception, if present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    return parse_retry_after_header(headers)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )):
        return True
    # APIStatusError is the catch-all for proxy / gateway failures
    # that aren't mapped to a more specific subclass — OpenRouter wraps
    # upstream 429 here because the proxy emits it, not OpenAI itself.
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 429 or (status is not None and 500 <= status < 600):
            return True
    return False


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
        max_tokens: int = 16384,
    ) -> None:
        kwargs: dict = {"api_key": api_key, "base_url": base_url, "timeout": _TIMEOUT}
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._base_url = base_url

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        kwargs: dict = {
            "messages": [_serialize_message(m) for m in messages],
            "max_tokens": self._max_tokens,
        }
        # model="" lets gateway auto-route (Cloudflare, etc.)
        if self._model:
            kwargs["model"] = self._model
        if tools:
            kwargs["tools"] = [_serialize_tool(t) for t in tools]
            kwargs["parallel_tool_calls"] = True
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        resp = await with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            is_retryable=_is_retryable,
            parse_retry_after=_parse_retry_after,
            provider="openai",
        )

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



# ── IR → OpenAI serialization ────────────────────────────────────────


def _serialize_message(msg: Message) -> dict:
    match msg.role:
        case Role.SYSTEM:
            return {"role": "system", "content": msg.text or ""}
        case Role.USER:
            return {"role": "user", "content": _serialize_parts(msg.parts)}
        case Role.ASSISTANT:
            m: dict = {"role": "assistant", "content": msg.text or ""}
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
