"""CLI event rendering — singledispatch, one handler per event type.

Event types here mirror the worker pipeline (`StageStarted/Done/Failed`,
`PageDone`) and the LLM agent loop (`LLMCall/Response`, `ToolResult`).
Chapter-level events from the deleted CLI ingest path
(`ChapterDownloaded/Skipped/Failed/Done`) are not handled — they no
longer exist.
"""

from __future__ import annotations

import traceback
from functools import singledispatch

from rich.console import Console

from typoon.runs.events import (
    Event, LLMCall, LLMResponse, PipelineError, StageDone, StageStarted,
    StageFailed, ToolResult,
)

console = Console()


@singledispatch
def render(event: Event) -> None:
    pass


@render.register
def _(event: StageStarted) -> None:
    console.print(f"    [dim]{event.stage}…[/]")


@render.register
def _(event: StageDone) -> None:
    console.print(f"    [green]✓[/] {event.stage}")


@render.register
def _(event: StageFailed) -> None:
    console.print(f"    [red]✗[/] {event.stage}: {event.error}")
    if event.error and event.error.__traceback__:
        console.print("".join(traceback.format_tb(event.error.__traceback__)), style="dim red")


@render.register
def _(event: PipelineError) -> None:
    console.print(f"    [red]error[/] [{event.stage}] {event.error}")
    if event.error and event.error.__traceback__:
        console.print("".join(traceback.format_tb(event.error.__traceback__)), style="dim red")


@render.register
def _(event: LLMCall) -> None:
    console.print(f"      [dim]{event.agent} t{event.turn}…[/]")


@render.register
def _(event: LLMResponse) -> None:
    status = "[green]✓[/]" if event.tool_calls > 0 else "[yellow]–[/]"
    console.print(f"      {status} {event.tool_calls} tools  {event.ms:.0f}ms")


@render.register
def _(event: ToolResult) -> None:
    console.print(f"        [dim]→ {event.tool}: {event.result[:80]}[/]")
