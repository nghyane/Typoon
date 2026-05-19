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


#
# Three target kinds in the v5 pipeline:
#   - chapter     for prepare + scan (pixel-level work, shared by every
#                 translation of the chapter)
#   - draft       for translate (LLM work scoped to (chapter, lang, glossary))
#   - translation for render of per-user edits (fallback render uses
#                 draft as target)
#
# `chapter_id` is filled in for every stage so subscribers can group
# progress by chapter regardless of which sub-target the worker is on.
# `draft_id` / `translation_id` are populated when the stage's natural
# target is one of those; UIs that care about a specific draft or
# translation filter on those.


@dataclass
class StageStarted(Event):
    chapter_id:     int = 0
    draft_id:       int = 0
    translation_id: int = 0
    stage:          str = ""


@dataclass
class StageDone(Event):
    chapter_id:     int = 0
    draft_id:       int = 0
    translation_id: int = 0
    stage:          str = ""


@dataclass
class PageDone(Event):
    chapter_id:     int = 0
    draft_id:       int = 0
    translation_id: int = 0
    stage:          str = ""
    page_index:     int = 0
    page_total:     int = 0


@dataclass
class StageFailed(Event):
    chapter_id:     int = 0
    draft_id:       int = 0
    translation_id: int = 0
    stage:          str = ""
    error:          Exception | None = None


class Hook:
    quit_requested: bool = False

    def on(self, event: Event) -> None:
        pass


class LoggingHook(Hook):
    """Prints pipeline events to the Python logger."""

    import logging as _logging
    _log = _logging.getLogger("typoon.pipeline")

    def _tag(self, ev: "Event") -> str:
        chapter_id     = getattr(ev, "chapter_id", 0)
        draft_id       = getattr(ev, "draft_id", 0)
        translation_id = getattr(ev, "translation_id", 0)
        parts = []
        if chapter_id:     parts.append(f"ch{chapter_id}")
        if draft_id:       parts.append(f"d{draft_id}")
        if translation_id: parts.append(f"t{translation_id}")
        return "/".join(parts) or "?"

    def on(self, event: Event) -> None:
        if isinstance(event, StageStarted):
            self._log.info("[%s] %s started", self._tag(event), event.stage)
        elif isinstance(event, StageDone):
            self._log.info("[%s] %s done", self._tag(event), event.stage)
        elif isinstance(event, StageFailed):
            self._log.error(
                "[%s] %s failed: %s", self._tag(event), event.stage, event.error,
            )
        elif isinstance(event, LLMCall):
            self._log.info("[llm] %s turn %d", event.agent, event.turn)
        elif isinstance(event, LLMResponse):
            self._log.info(
                "[llm] %s turn %d → %d ops (%.0fms)",
                event.agent, event.turn, event.tool_calls, event.ms,
            )


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
