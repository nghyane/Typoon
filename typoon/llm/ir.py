"""Provider-agnostic intermediate representation for LLM messages.

Port of crates/engine/src/llm/ir.rs + tool.rs + provider.rs.
Each provider adapter serializes these to its own wire format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


# ── Content parts ────────────────────────────────────────────────────


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


# ── Tool calling ─────────────────────────────────────────────────────


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


# ── Messages ─────────────────────────────────────────────────────────


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


# ── Provider response ────────────────────────────────────────────────


@dataclass(slots=True)
class CallResponse:
    """Response from a provider call."""

    tool_calls: list[ToolCallMsg] = field(default_factory=list)
    text: str | None = None


# ── Stream events ────────────────────────────────────────────────────


class StreamTruncatedError(RuntimeError):
    """Raised when a stream ends with finish_reason='length' (output truncated)."""


class StreamEventType(Enum):
    THINKING_DELTA = auto()
    TEXT_DELTA = auto()
    TOOL_CALL_START = auto()
    TOOL_CALL_DELTA = auto()
    TOOL_CALL_DONE = auto()
    DONE = auto()


@dataclass(slots=True)
class StreamEvent:
    """A single event from a streaming LLM response."""

    type: StreamEventType
    text: str = ""
    tool_index: int = 0
    tool_id: str = ""
    tool_name: str = ""


# ── Provider protocol ────────────────────────────────────────────────


@runtime_checkable
class Provider(Protocol):
    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse: ...

    def stream(self, messages: list[Message], tools: list[ToolDef]) -> AsyncIterator[StreamEvent]:
        """Yield streaming events. Default: not implemented."""
        ...
