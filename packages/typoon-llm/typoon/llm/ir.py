"""Provider-agnostic intermediate representation for LLM messages.

Each provider adapter serializes these to its own wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class ContentPart:
    """A content block within a message."""

    text: str | None = None
    image_data_uri: str | None = None

    @staticmethod
    def of_text(text: str) -> ContentPart:
        return ContentPart(text=text)

    @staticmethod
    def of_image(data_uri: str) -> ContentPart:
        return ContentPart(image_data_uri=data_uri)


@dataclass(slots=True)
class ToolCallMsg:
    """A tool call returned by the LLM."""

    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass(slots=True)
class ToolDef:
    """A tool definition exposed to the LLM."""

    name:        str
    description: str
    parameters:  dict[str, Any]  # JSON Schema


class ToolResponse:
    """Provider-agnostic tool response returned by handlers."""

    __slots__ = ("text", "image_data_uri")

    def __init__(self, text: str, image_data_uri: str | None = None) -> None:
        self.text = text
        self.image_data_uri = image_data_uri


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


@dataclass(slots=True)
class Message:
    """A message in the conversation."""

    role: Role
    parts: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCallMsg] = field(default_factory=list)
    tool_call_id: str | None = None  # only for TOOL_RESULT

    # ── Convenience constructors ──

    @staticmethod
    def system(text: str) -> Message:
        return Message(role=Role.SYSTEM, parts=[ContentPart.of_text(text)])

    @staticmethod
    def user_text(text: str) -> Message:
        return Message(role=Role.USER, parts=[ContentPart.of_text(text)])

    @staticmethod
    def user_parts(parts: list[ContentPart]) -> Message:
        return Message(role=Role.USER, parts=parts)

    @staticmethod
    def assistant(text: str | None = None, tool_calls: list[ToolCallMsg] | None = None) -> Message:
        parts = [ContentPart.of_text(text)] if text else []
        return Message(role=Role.ASSISTANT, parts=parts, tool_calls=tool_calls or [])

    @staticmethod
    def tool_result_text(tool_call_id: str, text: str) -> Message:
        return Message(
            role=Role.TOOL_RESULT,
            parts=[ContentPart.of_text(text)],
            tool_call_id=tool_call_id,
        )

    @staticmethod
    def tool_result_parts(tool_call_id: str, parts: list[ContentPart]) -> Message:
        return Message(role=Role.TOOL_RESULT, parts=parts, tool_call_id=tool_call_id)

    @property
    def text(self) -> str | None:
        """Extract first text content, if any."""
        for p in self.parts:
            if p.text is not None:
                return p.text
        return None


@dataclass(slots=True)
class CallResponse:
    """Response from a provider call."""

    tool_calls: list[ToolCallMsg] = field(default_factory=list)
    text: str | None = None


@runtime_checkable
class Provider(Protocol):
    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse: ...
