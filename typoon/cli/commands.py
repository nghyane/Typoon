"""CLI — user interaction only. Business logic lives in ProjectService."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..runs.events import Hook
from .events import render as render_event

app = typer.Typer(name="typoon", help="Manga translation pipeline.")
console = Console()


class ConsoleHook(Hook):
    def on(self, event) -> None:
        render_event(event)


# ── auth ──────────────────────────────────────────────────────────────


@app.command()
def auth(site: str = typer.Argument(..., help="Site to authenticate (e.g. comix.to)")):
    """Authenticate with a manga source site."""
    asyncio.run(_auth(site))


async def _auth(site: str) -> None:
    from ..sources.connectors import get_connectors
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
    target: str = typer.Argument(..., help="URL (new project) or slug (add to existing)"),
    url: str = typer.Argument(None, help="Manga URL (only when slug is first arg)"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    from_ch: float = typer.Option(0, "--from", help="First chapter (0 = ask)"),
    to_ch: float = typer.Option(0, "--to", help="Last chapter"),
    redo: str = typer.Option(None, "--redo", help="Re-run from: scan|translate|render"),
):
    """Download chapters from URL and translate.

    \b
    Examples:
      typoon pull https://comix.to/manga/solo-leveling/ -t vi
      typoon pull solo-leveling https://comix.to/manga/solo-leveling/ --from 10
    """
    asyncio.run(_pull(target, url, target_lang, from_ch, to_ch, redo))


async def _pull(target: str, url: str | None, target_lang: str, from_ch: float, to_ch: float, redo: str | None) -> None:
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        # Determine mode: new project or add to existing
        if url is not None:
            # typoon pull <slug> <url> — add to existing project
            slug = target
            await service.require_project(slug)  # validate slug exists
            info = await service.discover(url)
            console.print(f"\n[bold]{info.suggested_title}[/] — {len(info.chapters)} chapters")
            selected = _select_chapters(info.chapters, from_ch, to_ch)
            if not selected:
                return
            console.print(f"  {len(selected)} chapter(s) selected\n")
            await service.pull_more(slug, url, selected, ConsoleHook(), redo)
        else:
            # typoon pull <url> — new project or existing by source_url
            pull_url = target
            info = await service.discover(pull_url)
            console.print(f"\n[bold]{info.suggested_title}[/] — {len(info.chapters)} chapters ({info.suggested_lang})")
            selected = _select_chapters(info.chapters, from_ch, to_ch)
            if not selected:
                return
            console.print(f"  {len(selected)} chapter(s) selected\n")
            slug = await service.pull_new(info, pull_url, selected, target_lang, ConsoleHook(), redo)
            console.print(f"\n[dim]Project slug: [bold]{slug}[/][/]")
    finally:
        await service.close()


# ── add ───────────────────────────────────────────────────────────────


@app.command()
def add(
    target: str = typer.Argument(..., help="Folder path (new project) or slug (add to existing)"),
    folder: Path = typer.Argument(None, help="Chapter folder (only when slug is first arg)"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    source_lang: str = typer.Option("ko", "--source-lang", "-s"),
    name: str = typer.Option(None, "--name", "-n", help="Project name (default: folder name)"),
    redo: str = typer.Option(None, "--redo"),
):
    """Import local chapter images and translate.

    \b
    Examples:
      typoon add ./solo-leveling/ -s ko -t vi
      typoon add solo-leveling ./ch015/
    """
    asyncio.run(_add(target, folder, target_lang, source_lang, name, redo))


async def _add(target: str, folder: Path | None, target_lang: str, source_lang: str, name: str | None, redo: str | None) -> None:
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        if folder is not None:
            # typoon add <slug> <folder> — add to existing project
            slug = target
            await service.require_project(slug)
            src = Path(folder)
            if not src.is_dir():
                console.print(f"[red]Not a directory:[/] {src}")
                raise typer.Exit(2)
            await service.add_more(slug, src, ConsoleHook(), redo)
        else:
            # typoon add <folder> — new project (or existing by slug)
            src = Path(target)
            if not src.is_dir():
                console.print(f"[red]Not a directory:[/] {src}")
                raise typer.Exit(2)
            title = name or src.name
            slug = await service.add_new(src, title, source_lang, target_lang, ConsoleHook(), redo)
            console.print(f"\n[dim]Project slug: [bold]{slug}[/][/]")
    finally:
        await service.close()


# ── translate ─────────────────────────────────────────────────────────


@app.command()
def translate(
    slug: str = typer.Argument(..., help="Project slug"),
    from_ch: float = typer.Option(0, "--from", help="First chapter (0 = all pending)"),
    to_ch: float = typer.Option(0, "--to"),
    redo: str = typer.Option(None, "--redo"),
):
    """Translate pending chapters of a project.

    \b
    Examples:
      typoon translate solo-leveling
      typoon translate solo-leveling --from 5 --to 10
      typoon translate solo-leveling --redo translate
    """
    asyncio.run(_translate(slug, from_ch, to_ch, redo))


async def _translate(slug: str, from_ch: float, to_ch: float, redo: str | None) -> None:
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        proj = await service.require_project(slug)
        indices: list[float] | None = None
        if from_ch > 0 or to_ch > 0:
            all_chs = await service._db.get_all_chapters(proj["id"])
            lo = from_ch or all_chs[0]["idx"]
            hi = to_ch or all_chs[-1]["idx"]
            indices = [c["idx"] for c in all_chs if lo <= c["idx"] <= hi]
        await service.translate(slug, indices, ConsoleHook(), redo)
    finally:
        await service.close()


# ── status ────────────────────────────────────────────────────────────


@app.command()
def status(
    slug: str = typer.Argument(None, help="Project slug (omit for all)"),
):
    """Show project and chapter progress."""
    asyncio.run(_status(slug))


async def _status(slug: str | None) -> None:
    from ..adapters.project_service import ProjectService

    service = await ProjectService.open()
    try:
        all_status = await service.get_status(slug)
        if not all_status:
            msg = f"Project '{slug}' not found." if slug else "No projects yet. Run: typoon add <folder>"
            console.print(f"[dim]{msg}[/]")
            return

        for proj in all_status:
            _print_project(proj)
    finally:
        await service.close()


def _print_project(proj) -> None:
    done    = sum(1 for c in proj.chapters if c.status == "done")
    total   = len(proj.chapters)
    console.print(
        f"\n[bold]{proj.title}[/]  [dim]{proj.slug}[/]  "
        f"{proj.source_lang}→{proj.target_lang}  "
        f"[green]{done}[/]/[dim]{total}[/] chapters"
    )
    if not proj.chapters:
        console.print("  [dim]No chapters[/]")
        return
    t = Table(show_header=False, box=None, padding=(0, 1))
    for ch in proj.chapters:
        icon = {"done": "[green]✓[/]", "translating": "[yellow]⟳[/]",
                "error": "[red]✗[/]", "pending": "[dim]○[/]",
                "downloading": "[cyan]↓[/]"}.get(ch.status, "?")
        info = f"[dim]{ch.render_count} pages[/]" if ch.render_count else ""
        t.add_row(icon, f"ch{ch.idx:03.0f}", f"[dim]{ch.status}[/]", info)
    console.print(t)


# ── Shared UI helpers ─────────────────────────────────────────────────


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
