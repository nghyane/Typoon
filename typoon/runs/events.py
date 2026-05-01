"""Run events emitted by LLM and future stage orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Event:
    ts: float = field(default_factory=time.time, repr=False)


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
    agent: str = ""
    turn: int = 0
    tool_name: str = ""


@dataclass
class ToolResult(Event):
    agent: str = ""
    turn: int = 0
    tool: str = ""
    result: str = ""


@dataclass
class TranslationWindow(Event):
    """One translate_window call started/done."""
    chapter: float = 0.0
    window: int = 0
    total_windows: int = 0
    active_keys: int = 0
    status: str = "started"  # started | done | retry
    ms: float = 0.0
    turns: int = 0
    missing: int = 0


@dataclass
class PipelineError(Event):
    stage: str = ""
    error: Exception | None = None


# ── Pipeline stage events ─────────────────────────────────────────────


@dataclass
class ChapterSkipped(Event):
    idx:    float = 0.0
    reason: str = ""


@dataclass
class ChapterDownloaded(Event):
    idx:        float = 0.0
    page_count: int = 0


@dataclass
class StageStarted(Event):
    idx:   float = 0.0
    stage: str = ""


@dataclass
class StageDone(Event):
    idx:   float = 0.0
    stage: str = ""


@dataclass
class ChapterDone(Event):
    idx:          float = 0.0
    bubble_count: int = 0
    render_dir:   str = ""


@dataclass
class ChapterFailed(Event):
    idx:   float = 0.0
    stage: str = ""
    error: Exception | None = None


@dataclass
class StageFailed(Event):
    idx:   float = 0.0
    stage: str = ""
    error: Exception | None = None


class Hook:
    quit_requested: bool = False

    def on(self, event: Event) -> None:
        pass


class CompositeHook(Hook):
    def __init__(self, *hooks: Hook) -> None:
        self._hooks = hooks

    def on(self, event: Event) -> None:
        for hook in self._hooks:
            hook.on(event)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class CompositeSink:
    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def emit(self, event: Event) -> None:
        for sink in self._sinks:
            sink.emit(event)


class HookAdapter(Hook):
    def __init__(self, sink: EventSink) -> None:
        self._sink = sink

    def on(self, event: Event) -> None:
        self._sink.emit(event)
