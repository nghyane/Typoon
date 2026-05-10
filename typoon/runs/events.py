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
class ModelsUnloaded(Event):
    stage: str = ""


@dataclass
class PipelineError(Event):
    stage: str = ""
    error: Exception | None = None


# ── Pipeline stage events ─────────────────────────────────────────────


@dataclass
class StageStarted(Event):
    chapter_id: int = 0
    project_id: int = 0
    stage:      str = ""


@dataclass
class StageDone(Event):
    chapter_id: int = 0
    project_id: int = 0
    stage:      str = ""


@dataclass
class PageDone(Event):
    chapter_id: int = 0
    project_id: int = 0
    stage:      str = ""
    page_index: int = 0
    page_total: int = 0


@dataclass
class StageFailed(Event):
    chapter_id: int = 0
    project_id: int = 0
    stage:      str = ""
    error:      Exception | None = None


class Hook:
    quit_requested: bool = False

    def on(self, event: Event) -> None:
        pass


class LoggingHook(Hook):
    """Prints pipeline events to the Python logger."""

    import logging as _logging
    _log = _logging.getLogger("typoon.pipeline")

    def on(self, event: Event) -> None:
        if isinstance(event, StageStarted):
            self._log.info("[ch%d] %s started", event.chapter_id, event.stage)
        elif isinstance(event, StageDone):
            self._log.info("[ch%d] %s done", event.chapter_id, event.stage)
        elif isinstance(event, StageFailed):
            self._log.error("[ch%d] %s failed: %s", event.chapter_id, event.stage, event.error)
        elif isinstance(event, LLMCall):
            self._log.info("[llm] %s turn %d", event.agent, event.turn)
        elif isinstance(event, LLMResponse):
            self._log.info("[llm] %s turn %d → %d ops (%.0fms)", event.agent, event.turn, event.tool_calls, event.ms)


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
