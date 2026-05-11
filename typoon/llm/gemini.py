"""Google Gemini provider — native SDK adapter."""

from __future__ import annotations

import json

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from ._retry import parse_retry_after_header, with_retry
from .errors import TransientCredentialError, UpstreamUnavailable
from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Role,
    ToolCallMsg,
    ToolDef,
)


def _parse_retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    return parse_retry_after_header(headers)


def _is_retryable(exc: BaseException) -> bool:
    # google-genai surfaces 4xx as ClientError, 5xx as ServerError.
    # Retry 429 (rate limit) and any 5xx transient.
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429 or (code is not None and 500 <= code < 600):
            return True
    # httpx-level transport failures bubble up unchanged when the SDK
    # cannot even reach the server.
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException)):
        return True
    return False


class GeminiProvider:
    """Google Gemini native provider."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        system_instruction, contents = _build_contents(messages)

        config: dict = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tools:
            config["tools"] = [_build_tools(tools)]

        try:
            resp = await with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config),
                ),
                is_retryable=_is_retryable,
                parse_retry_after=_parse_retry_after,
                provider="gemini",
            )
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403):
                raise TransientCredentialError(str(exc)) from exc
            raise
        except genai_errors.ServerError as exc:
            raise UpstreamUnavailable(str(exc)) from exc

        tool_calls: list[ToolCallMsg] = []
        text_parts: list[str] = []

        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCallMsg(
                        id=fc.id or f"call_{fc.name}",
                        name=fc.name,
                        arguments=json.dumps(fc.args) if fc.args else "{}",
                    ))
                elif part.text:
                    text_parts.append(part.text)

        text = "\n".join(text_parts) if text_parts else None
        return CallResponse(tool_calls=tool_calls, text=text)


# ── IR → Gemini serialization ────────────────────────────────────────


def _build_contents(
    messages: list[Message],
) -> tuple[str | None, list[types.Content]]:
    """Convert IR messages to Gemini contents. Returns (system_instruction, contents)."""
    system: str | None = None
    contents: list[types.Content] = []

    for msg in messages:
        match msg.role:
            case Role.SYSTEM:
                system = msg.text
            case Role.USER:
                contents.append(types.Content(
                    role="user",
                    parts=_serialize_parts(msg.parts),
                ))
            case Role.ASSISTANT:
                parts: list[types.Part] = []
                if msg.text:
                    parts.append(types.Part(text=msg.text))
                for tc in msg.tool_calls:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            id=tc.id, name=tc.name, args=args,
                        ),
                    ))
                contents.append(types.Content(role="model", parts=parts))
            case Role.TOOL_RESULT:
                text = "\n".join(p.text for p in msg.parts if p.text)
                name = _find_tool_name(messages, msg.tool_call_id)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            id=msg.tool_call_id,
                            name=name,
                            response={"result": text},
                        ),
                    )],
                ))

    return system, contents


def _serialize_parts(parts: list[ContentPart]) -> list[types.Part]:
    result: list[types.Part] = []
    for p in parts:
        if p.text is not None:
            result.append(types.Part(text=p.text))
        if p.image_data_uri is not None:
            data_uri = p.image_data_uri
            for prefix, mime in [
                ("data:image/jpeg;base64,", "image/jpeg"),
                ("data:image/png;base64,", "image/png"),
            ]:
                if data_uri.startswith(prefix):
                    import base64
                    raw = base64.b64decode(data_uri[len(prefix):])
                    result.append(types.Part(
                        inline_data=types.Blob(mime_type=mime, data=raw),
                    ))
                    break
            else:
                result.append(types.Part(text="[unsupported image format]"))
    return result


def _build_tools(tools: list[ToolDef]) -> types.Tool:
    """Build a single Gemini Tool with all function declarations."""
    declarations = []
    for t in tools:
        params = _strip_additional_properties(t.parameters)
        declarations.append(types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=params,
        ))
    return types.Tool(function_declarations=declarations)


def _strip_additional_properties(node: dict | list | object) -> dict | list | object:
    """Remove additionalProperties recursively — Gemini doesn't support it."""
    if isinstance(node, dict):
        return {k: _strip_additional_properties(v) for k, v in node.items() if k != "additionalProperties"}
    if isinstance(node, list):
        return [_strip_additional_properties(item) for item in node]
    return node


def _find_tool_name(messages: list[Message], tool_call_id: str | None) -> str:
    """Look up the tool name for a tool_call_id from prior assistant messages."""
    if tool_call_id is None:
        return "unknown"
    for msg in messages:
        for tc in msg.tool_calls:
            if tc.id == tool_call_id:
                return tc.name
    return "unknown"
