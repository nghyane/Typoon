"""App-level events and sinks.

Legacy Hook is bridged via HookAdapter.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..events import Event, Hook  # noqa: F401


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
