"""CLI — user interaction only. Business logic lives in ProjectService."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..runs.events import (
    ChapterDone, ChapterDownloaded, ChapterFailed,
    ChapterSkipped, Hook, StageDone, StageStarted,
)

app = typer.Typer(name="typoon", help="Manga translation pipeline.")
console = Console()


class ConsoleHook(Hook):
    """Renders pipeline events to the terminal."""

    def on(self, event) -> None:
        match event:
            case ChapterDownloaded():
                console.print(f"  [cyan]↓[/] ch{event.idx:03.0f}  {event.page_count} pages")
            case ChapterSkipped():
                console.print(f"  [dim]–[/] ch{event.idx:03.0f}  {event.reason}")
            case StageStarted():
                console.print(f"    [dim]{event.stage}…[/]", end="")
            case StageDone():
                console.print(" [green]✓[/]")
            case ChapterDone():
                console.print(f"  [green]✓[/] ch{event.idx:03.0f}  {event.bubble_count} bubbles")
                console.print(f"    [dim]{event.render_dir}[/]")
            case ChapterFailed():
                console.print(f"  [red]✗[/] ch{event.idx:03.0f}  {event.stage}: {event.error}")


# ── auth ──────────────────────────────────────────────────────────────


@app.command()
def auth(site: str = typer.Argument(..., help="Site to authenticate (e.g. comix.to)")):
    """Authenticate with a manga source site."""
    asyncio.run(_auth(site))


async def _auth(site: str):
    from ..adapters.connectors import get_connectors
    connector = next((c for c in get_connectors() if site in c.site_name or c.site_name in site), None)
    if connector is None:
        console.print(f"[red]Unknown site:[/] {site}")
        raise typer.Exit(1)
    console.print(f"[yellow]Opening browser for {connector.site_name}…[/]")
    await connector.authenticate()
    console.print(f"[green]✓[/] {connector.site_name} — authenticated")


# ── pull ──────────────────────────────────────────────────────────────


@app.command()
def pull(
    url: str = typer.Argument(..., help="Manga or chapter URL"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    from_ch: float = typer.Option(0, "--from", help="First chapter (0 = ask interactively)"),
    to_ch: float = typer.Option(0, "--to", help="Last chapter"),
    redo: str = typer.Option(None, "--redo", help="Re-run from stage: scan|translate|render"),
):
    """Download chapters from a URL and translate."""
    asyncio.run(_pull(url, target_lang, from_ch, to_ch, redo))


async def _pull(url: str, target_lang: str, from_ch: float, to_ch: float, redo: str | None):
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        info = await service.discover(url)
        console.print(f"\n[bold]{info.suggested_title}[/] — {len(info.chapters)} chapters ({info.suggested_lang})")

        selected = _select_chapters(info.chapters, from_ch, to_ch)
        if not selected:
            console.print("[yellow]No chapters selected.[/]")
            return
        console.print(f"  {len(selected)} chapter(s) selected\n")

        await service.pull(url, selected, target_lang, ConsoleHook(), redo)
    finally:
        await service.close()


# ── add ───────────────────────────────────────────────────────────────


@app.command()
def add(
    folder: Path = typer.Argument(..., help="Local folder with chapter images"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    source_lang: str = typer.Option("ko", "--source-lang", "-s"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (default: folder name)"),
    redo: str = typer.Option(None, "--redo"),
):
    """Import local chapter images and translate."""
    asyncio.run(_add(folder, target_lang, source_lang, project, redo))


async def _add(folder: Path, target_lang: str, source_lang: str, project_name: str | None, redo: str | None):
    from ..adapters.project_service import ProjectService

    if not folder.is_dir():
        console.print(f"[red]Not a directory:[/] {folder}")
        raise typer.Exit(2)

    service = await ProjectService.open()
    try:
        await service.add(folder, project_name or folder.name, source_lang, target_lang, ConsoleHook(), redo)
    finally:
        await service.close()


# ── translate ─────────────────────────────────────────────────────────


@app.command()
def translate(
    project: str = typer.Argument(None, help="Project slug or name (omit for all pending)"),
    chapter: float = typer.Option(None, "--chapter", "-c"),
    redo: str = typer.Option(None, "--redo"),
):
    """Translate pending chapters."""
    asyncio.run(_translate_cmd(project, chapter, redo))


async def _translate_cmd(project_name: str | None, chapter_num: float | None, redo: str | None):
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        all_status = await service.get_status()
        projects = all_status if not project_name else [
            p for p in all_status if p["slug"] == project_name or p["title"] == project_name
        ]
        for proj in projects:
            indices = ([chapter_num] if chapter_num is not None
                       else [c["idx"] for c in proj["chapters"] if c["status"] == "pending"])
            if indices:
                await service.translate(proj["id"], indices, ConsoleHook(), redo)
    finally:
        await service.close()


# ── status ────────────────────────────────────────────────────────────


@app.command()
def status(
    project: str = typer.Argument(None, help="Project slug or name"),
):
    """Show all projects and chapter progress."""
    asyncio.run(_status(project))


async def _status(project_name: str | None):
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        all_status = await service.get_status()
        if not all_status:
            console.print("[dim]No projects yet. Run: typoon pull <url>[/]")
            return
        for proj in all_status:
            if project_name and proj["slug"] != project_name and proj["title"] != project_name:
                continue
            console.print(f"\n[bold]{proj['title']}[/] [dim]({proj['slug']})[/]  {proj['source_lang']} → {proj['target_lang']}")
            if not proj["chapters"]:
                console.print("  [dim]No chapters imported[/]")
                continue
            t = Table(show_header=False, box=None, padding=(0, 1))
            for ch in proj["chapters"]:
                icon = {"done": "[green]✓[/]", "translating": "[yellow]⟳[/]",
                        "error": "[red]✗[/]", "pending": "[dim]○[/]"}.get(ch["status"], "?")
                info = f"[dim]{ch['render_count']} pages → {ch['cp'].render}[/]" if ch["render_count"] else ""
                t.add_row(icon, f"ch{ch['idx']:03.0f}", ch["status"], info)
            console.print(t)
    finally:
        await service.close()


# ── UI helpers ────────────────────────────────────────────────────────


def _select_chapters(chapters, from_ch: float, to_ch: float) -> list:
    if from_ch > 0 or to_ch > 0:
        lo = from_ch or chapters[0].number
        hi = to_ch or chapters[-1].number
        return [c for c in chapters if lo <= c.number <= hi]
    console.print(f"  Available: ch{chapters[0].number:.0f} – ch{chapters[-1].number:.0f}")
    raw = typer.prompt("  Select chapters (e.g. 1-5 or 3)").strip()
    if "-" in raw:
        lo, hi = (float(x.strip()) for x in raw.split("-", 1))
    else:
        lo = hi = float(raw)
    return [c for c in chapters if lo <= c.number <= hi]
