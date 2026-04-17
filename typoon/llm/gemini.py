"""Google Gemini provider — native SDK adapter.

Serializes IR → Gemini format, calls via `google-genai` SDK,
parses response back to CallResponse IR. Supports streaming.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

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

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )

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

    async def stream(self, messages: list[Message], tools: list[ToolDef]) -> AsyncIterator[StreamEvent]:
        system_instruction, contents = _build_contents(messages)

        config: dict = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tools:
            config["tools"] = [_build_tools(tools)]

        tc_counter = 0
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        ):
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for part in chunk.candidates[0].content.parts:
                if part.thought:
                    yield StreamEvent(type=StreamEventType.THINKING_DELTA, text=part.text or "")
                elif part.function_call:
                    fc = part.function_call
                    tid = fc.id or f"call_{fc.name}"
                    args = json.dumps(fc.args) if fc.args else "{}"
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_index=tc_counter,
                        tool_id=tid,
                        tool_name=fc.name,
                    )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        tool_index=tc_counter,
                        text=args,
                    )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DONE,
                        tool_index=tc_counter,
                        tool_id=tid,
                        tool_name=fc.name,
                        text=args,
                    )
                    tc_counter += 1
                elif part.text:
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=part.text)

        yield StreamEvent(type=StreamEventType.DONE)


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
        declarations.append(types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        ))
    return types.Tool(function_declarations=declarations)


def _find_tool_name(messages: list[Message], tool_call_id: str | None) -> str:
    """Look up the tool name for a tool_call_id from prior assistant messages."""
    if tool_call_id is None:
        return "unknown"
    for msg in messages:
        for tc in msg.tool_calls:
            if tc.id == tool_call_id:
                return tc.name
    return "unknown"
