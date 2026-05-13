"""OpenAI provider — native SDK adapter.

Serializes IR → OpenAI chat format, calls via `openai` SDK,
parses response back to CallResponse IR. Supports streaming.
"""

from __future__ import annotations

import openai

from ._retry import parse_retry_after_header, with_retry
from .errors import OperatorActionRequired, UpstreamUnavailable
from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Role,
    ToolCallMsg,
    ToolDef,
)

_TIMEOUT = 300

# Substrings the proxy emits when its credential pool is empty / a key
# was revoked. Distinct from a model's own auth pushback (e.g. plain
# "unauthorized") so we don't reclassify legit user-config errors as
# transient. Bifrost: `no available credential for provider`. Upstream
# Codex/LunchDock: `token_invalidated`, `Your authentication token has
# been invalidated`.
_CREDENTIAL_HINTS = (
    "token_invalidated",
    "no available credential",
    "credential has been invalidated",
    "authentication token has been invalidated",
)

# Substrings inside a 5xx body that mean retrying is hopeless — the
# operator must change config (or the routing group) before any
# request can succeed. Distinct from a real upstream outage which is
# expected to clear on its own.
_OPERATOR_5XX_HINTS = (
    "model_not_found",
    "model not found",
    "no available channel",
    "no available channels",
    "distributor",                # Packy: "无可用渠道 (distributor)"
    "billing_hard_limit",
    "insufficient_quota",
    "account is suspended",
    "region not supported",
)


def _looks_like_credential_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(hint.lower() in msg for hint in _CREDENTIAL_HINTS)


def _looks_like_operator_5xx(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(hint.lower() in msg for hint in _OPERATOR_5XX_HINTS)


def _classify_status_error(exc: openai.APIStatusError) -> Exception | None:
    """Translate a non-retryable APIStatusError into our taxonomy.

    Returns the wrapped exception to raise, or None to let the original
    propagate untouched (caller path: bugs / 4xx user errors).
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return OperatorActionRequired(str(exc))
    if status is not None and 500 <= status < 600:
        # 5xx that survived `with_retry`. Split into:
        #   • operator: body says model/credential/quota — retrying
        #     forever is pointless without a config change.
        #   • transient: anything else is treated as upstream down,
        #     waits on backoff and retries on its own.
        if _looks_like_operator_5xx(exc) or _looks_like_credential_failure(exc):
            return OperatorActionRequired(str(exc))
        return UpstreamUnavailable(str(exc))
    if _looks_like_credential_failure(exc):
        # Some proxies surface credential issues as 200/4xx with a JSON
        # error body — catch that path too.
        return OperatorActionRequired(str(exc))
    return None


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
        api_kind: str = "chat",
    ) -> None:
        kwargs: dict = {"api_key": api_key, "base_url": base_url, "timeout": _TIMEOUT}
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._base_url = base_url
        if api_kind not in ("chat", "responses"):
            raise ValueError(f"api_kind must be 'chat' or 'responses', got {api_kind!r}")
        self._api_kind = api_kind

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        if self._api_kind == "responses":
            return await self._call_responses(messages, tools)
        return await self._call_chat(messages, tools)

    async def _call_chat(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
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

        try:
            resp = await with_retry(
                lambda: self._client.chat.completions.create(**kwargs),
                is_retryable=_is_retryable,
                parse_retry_after=_parse_retry_after,
                provider="openai",
            )
        except openai.APIStatusError as exc:
            wrapped = _classify_status_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

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

    async def _call_responses(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        """OpenAI /v1/responses path.

        The Responses API uses `input` (not `messages`), expects multimodal
        content as `input_text` / `input_image` parts (not `text` / `image_url`),
        and serializes tools/tool_calls differently from chat completions.

        Always streams: some OpenAI-compatible proxies (e.g. Packy/Bifrost)
        never populate `response.output` on non-stream calls — text only
        arrives through SSE deltas. Streaming is also the documented
        recommendation upstream, so we standardize on it.
        """
        input_items, instructions = _build_responses_input(messages)

        kwargs: dict = {
            "input": input_items,
            "max_output_tokens": self._max_tokens,
            "stream": True,
        }
        if self._model:
            kwargs["model"] = self._model
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = [_serialize_tool_responses(t) for t in tools]
            kwargs["parallel_tool_calls"] = True
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        try:
            stream = await with_retry(
                lambda: self._client.responses.create(**kwargs),
                is_retryable=_is_retryable,
                parse_retry_after=_parse_retry_after,
                provider="openai",
            )
        except openai.APIStatusError as exc:
            wrapped = _classify_status_error(exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

        text_parts: list[str] = []
        # function_call args arrive as streamed deltas keyed by item_id.
        fc_state: dict[str, dict] = {}

        async for ev in stream:
            etype = getattr(ev, "type", "") or ""
            if etype == "response.output_text.delta":
                delta = getattr(ev, "delta", None)
                if delta:
                    text_parts.append(delta)
            elif etype == "response.output_item.added":
                item = getattr(ev, "item", None)
                if item is not None and getattr(item, "type", None) == "function_call":
                    fc_state[item.id] = {
                        "call_id": getattr(item, "call_id", None) or item.id,
                        "name": getattr(item, "name", "") or "",
                        "arguments": getattr(item, "arguments", "") or "",
                    }
            elif etype == "response.function_call_arguments.delta":
                item_id = getattr(ev, "item_id", None)
                delta = getattr(ev, "delta", "") or ""
                if item_id and item_id in fc_state:
                    fc_state[item_id]["arguments"] += delta
            elif etype == "response.completed":
                # Some proxies also drop a final assembled response here;
                # nothing to do — our deltas already captured the content.
                pass

        tool_calls = [
            ToolCallMsg(id=st["call_id"], name=st["name"], arguments=st["arguments"])
            for st in fc_state.values()
        ]
        text = "".join(text_parts) or None
        if not tool_calls and not text:
            raise RuntimeError(
                f"OpenAI Responses stream produced no content. model={self._model}"
            )
        return CallResponse(tool_calls=tool_calls, text=text)



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


# ── IR → OpenAI Responses serialization ──────────────────────────────


def _build_responses_input(messages: list[Message]) -> tuple[list[dict], str]:
    """Translate chat-shaped IR into Responses `input` items + `instructions`.

    Differences vs chat completions:
      - System messages collapse into a top-level `instructions` string.
      - User/assistant messages use `input_text` / `input_image` parts.
      - Tool calls are top-level `function_call` items, tool results are
        `function_call_output` items keyed by `call_id`.
    """
    instructions_parts: list[str] = []
    items: list[dict] = []
    for m in messages:
        match m.role:
            case Role.SYSTEM:
                if m.text:
                    instructions_parts.append(m.text)
            case Role.USER:
                items.append({
                    "role": "user",
                    "content": _serialize_parts_responses(m.parts, input=True),
                })
            case Role.ASSISTANT:
                if m.text:
                    items.append({
                        "role": "assistant",
                        "content": _serialize_parts_responses(m.parts, input=False),
                    })
                for tc in m.tool_calls:
                    items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })
            case Role.TOOL_RESULT:
                # Responses API expects the result body as a plain string.
                # Collapse any multipart text into one; images in tool
                # results are not part of the spec.
                body = "\n".join(p.text for p in m.parts if p.text is not None)
                items.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": body,
                })

    return items, "\n\n".join(instructions_parts)


def _serialize_parts_responses(parts: list[ContentPart], *, input: bool) -> list[dict]:
    text_type = "input_text" if input else "output_text"
    out: list[dict] = []
    for p in parts:
        if p.text is not None:
            out.append({"type": text_type, "text": p.text})
        if p.image_data_uri is not None:
            # Responses API: image goes under `input_image.image_url` as a
            # plain URL string (data: URIs accepted). `detail` is optional.
            out.append({
                "type": "input_image",
                "image_url": p.image_data_uri,
                "detail": "low",
            })
    return out


def _serialize_tool_responses(tool: ToolDef) -> dict:
    """Responses API tool schema — flat shape, not nested under 'function'."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
