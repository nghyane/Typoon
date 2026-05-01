"""CLI event rendering — singledispatch, one handler per event type."""

from __future__ import annotations

from functools import singledispatch

from rich.console import Console

from typoon.runs.events import (
    ChapterDone, ChapterDownloaded, ChapterFailed,
    ChapterSkipped, Event, StageDone, StageStarted, StageFailed,
)

console = Console()


@singledispatch
def render(event: Event) -> None:
    """Default: ignore unknown events."""


@render.register
def _(event: ChapterDownloaded) -> None:
    console.print(f"  [cyan]↓[/] ch{event.idx:03.0f}  {event.page_count} pages")


@render.register
def _(event: ChapterSkipped) -> None:
    console.print(f"  [dim]–[/] ch{event.idx:03.0f}  {event.reason}")


@render.register
def _(event: StageStarted) -> None:
    console.print(f"    [dim]{event.stage}…[/]", end="")


@render.register
def _(event: StageDone) -> None:
    console.print(" [green]✓[/]")


@render.register
def _(event: StageFailed) -> None:
    console.print(f" [red]✗[/] {event.error}")


@render.register
def _(event: ChapterDone) -> None:
    console.print(f"  [green]✓[/] ch{event.idx:03.0f}  {event.bubble_count} bubbles")
    console.print(f"    [dim]{event.render_dir}[/]")


@render.register
def _(event: ChapterFailed) -> None:
    console.print(f"  [red]✗[/] ch{event.idx:03.0f}  {event.stage}: {event.error}")
