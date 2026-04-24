"""Events and sinks.

All event types + Hook (legacy callback) + EventSink (new protocol).
EventSink is the modern interface; Hook bridges to legacy code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Base ──────────────────────────────────────────────────────────


@dataclass
class Event:
    ts: float = field(default_factory=time.time, repr=False)


# ── Scan ──────────────────────────────────────────────────────────


@dataclass
class PageScanned(Event):
    page: int = 0
    total: int = 0
    bubbles: int = 0
    det_ms: float = 0
    ocr_ms: float = 0


# ── Translation ───────────────────────────────────────────────────


@dataclass
class TranslateStart(Event):
    total_bubbles: int = 0


@dataclass
class TranslationReady(Event):
    """Emitted after translate, before erase. Consumers can show text immediately."""
    pages: int = 0
    translated: int = 0
    total: int = 0


@dataclass
class TranslateDone(Event):
    translated: int = 0
    total: int = 0
    turns: int = 0


@dataclass
class LLMCall(Event):
    agent: str = ""
    turn: int = 0


@dataclass
class LLMThinking(Event):
    agent: str = ""
    turn: int = 0
    delta: str = ""


@dataclass
class LLMText(Event):
    agent: str = ""
    turn: int = 0
    delta: str = ""


@dataclass
class LLMResponse(Event):
    agent: str = ""
    turn: int = 0
    tool_calls: int = 0
    ms: float = 0


@dataclass
class ToolCallStart(Event):
    """Emitted when a tool call begins streaming (before arguments arrive)."""
    agent: str = ""
    turn: int = 0
    tool_name: str = ""


@dataclass
class ToolResult(Event):
    agent: str = ""
    turn: int = 0
    tool: str = ""
    result: str = ""


# ── Erase + Render ──────────────────────────────────────────────


@dataclass
class PageErased(Event):
    page: int = 0
    total: int = 0
    ms: float = 0


@dataclass
class PageRendered(Event):
    page: int = 0
    total: int = 0


# ── Chapter / Series ────────────────────────────────────────────


@dataclass
class ChapterStart(Event):
    project_id: int = 0
    chapter: float = 0
    pages: int = 0


@dataclass
class ChapterDone(Event):
    chapter: float = 0
    pages: int = 0
    bubbles: int = 0
    elapsed: float = 0


@dataclass
class ChapterSkipped(Event):
    chapter: float = 0
    reason: str = ""


@dataclass
class SeriesProgress(Event):
    """Emitted after each chapter completes/skips in a series run."""
    done: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0


@dataclass
class ModelsUnloaded(Event):
    stage: str = ""  # "scan" or "erase"


# ── Knowledge ─────────────────────────────────────────────────────


@dataclass
class KnowledgeStart(Event):
    chapter: int = 0
    pairs: int = 0


@dataclass
class KnowledgeDone(Event):
    chapter: int = 0
    turns: int = 0


# ── Errors ────────────────────────────────────────────────────────


@dataclass
class PipelineError(Event):
    stage: str = ""
    error: Exception | None = None


# ── Hook (legacy callback interface) ──────────────────────────────


class Hook:
    """Override on() to handle events. Default: ignore all."""

    quit_requested: bool = False

    def on(self, event: Event) -> None:
        pass


class CompositeHook(Hook):
    """Fan out to multiple hooks."""

    def __init__(self, *hooks: Hook) -> None:
        self._hooks = hooks

    def on(self, event: Event) -> None:
        for h in self._hooks:
            h.on(event)


# ── EventSink (modern protocol) ───────────────────────────────────


@runtime_checkable
class EventSink(Protocol):
    """Receives pipeline events. Implement to build a UI."""

    def emit(self, event: Event) -> None: ...


class CompositeSink:
    """Fan out to multiple sinks."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def emit(self, event: Event) -> None:
        for s in self._sinks:
            s.emit(event)


class HookAdapter(Hook):
    """Bridge: EventSink → Hook so workflows can pass sinks to legacy code."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink

    def on(self, event: Event) -> None:
        self._sink.emit(event)
