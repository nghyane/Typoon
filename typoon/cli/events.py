"""CLI event rendering — singledispatch, one handler per event type."""

from __future__ import annotations

import traceback
from functools import singledispatch

from rich.console import Console

from typoon.runs.events import (
    ChapterDone, ChapterDownloaded, ChapterFailed, ChapterSkipped,
    Event, LLMCall, LLMResponse, PipelineError, StageDone, StageStarted,
    StageFailed, ToolResult,
)

console = Console()


@singledispatch
def render(event: Event) -> None:
    pass


@render.register
def _(event: ChapterDownloaded) -> None:
    console.print(f"  [cyan]↓[/] ch{event.chapter_idx:.4g}  {event.page_count} pages")


@render.register
def _(event: ChapterSkipped) -> None:
    console.print(f"  [dim]–[/] ch{event.chapter_idx:.4g}  {event.reason}")


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
def _(event: ChapterDone) -> None:
    console.print(f"  [green]✓[/] ch{event.chapter_idx:.4g}  {event.bubble_count} pages rendered")
    console.print(f"    [dim]{event.render_dir}[/]")


@render.register
def _(event: ChapterFailed) -> None:
    console.print(f"  [red]✗[/] ch{event.chapter_idx:.4g}  {event.stage}: {event.error}")
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
