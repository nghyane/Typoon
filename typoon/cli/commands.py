"""CLI commands — typer app + interactive TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import typer
from rich.console import Console

from .output import SingleFileSource, save_pages
from .pipeline import run_pipeline, translate_cli
from .resolve import _get_connectors
from .utils import has_chapter_subdirs, has_images, is_url

app = typer.Typer(name="typoon", invoke_without_command=True)
console = Console()


@app.command()
def status():
    """Show connectors and authentication status."""
    console.print("[bold]Connectors[/]\n")
    for c in _get_connectors():
        if c.is_authenticated():
            console.print(f"  [green]●[/] {c.site_name:20s} authenticated")
        else:
            console.print(f"  [red]○[/] {c.site_name:20s} [dim]run:[/] typoon auth {c.site_name}")
    console.print()


@app.command()
def auth(site: str = typer.Argument(..., help="Site to authenticate (e.g. comix.to)")):
    """Authenticate with a manga source (opens browser if needed)."""
    asyncio.run(_auth(site))


async def _auth(site: str):
    connector = next((c for c in _get_connectors() if site in c.site_name or c.site_name in site), None)
    if connector is None:
        console.print(f"[red]Unknown site: {site}[/]")
        raise typer.Exit(1)
    console.print(f"[yellow]Solving Cloudflare challenge for {connector.site_name}…[/]")
    console.print("[dim]Complete the challenge in the browser window.[/]")
    await connector.authenticate()
    console.print(f"[green]✓[/] {connector.site_name} — authenticated")


@app.command()
def detect(path: Path = typer.Argument(..., help="Image file or folder")):
    """Vision only: detect → merge → OCR → erase."""
    from ..cli.hook import RichHook
    from ..adapters.local_source import LocalSource
    from ..engine import Engine

    engine, config, paths = Engine.from_config()
    paths.output.mkdir(exist_ok=True)
    hook = RichHook()

    source = LocalSource(path) if path.is_dir() else SingleFileSource(path)
    pages, images = engine.preprocess(source, hook=hook)
    engine.erase(pages, images, hook=hook)

    for page in pages:
        if page.erased is not None:
            out = paths.output / f"erased_p{page.index}.png"
            cv2.imwrite(str(out), cv2.cvtColor(page.erased, cv2.COLOR_RGB2BGR))
            console.print(f"  Saved: [cyan]{out}[/]")


@app.command()
def translate(
    input: str = typer.Argument(..., help="URL or local path"),
    source_lang: str = typer.Option("", "--source-lang", "-s"),
    target_lang: str = typer.Option("", "--target-lang", "-t"),
    from_ch: float = typer.Option(0, "--from"),
    to_ch: float = typer.Option(0, "--to"),
    force: bool = typer.Option(False, "--force"),
):
    """Translate manga — accepts URL or local path."""
    asyncio.run(translate_cli(input, source_lang, target_lang, from_ch, to_ch, force))


@app.command()
def clean(
    older_than: int = typer.Option(30, "--older-than", "-d", help="Remove cached images older than N days"),
    project: str = typer.Option(None, "--project", "-p", help="Only clean this project by slug"),
):
    """Remove cached source images older than N days to free disk space."""
    from ..paths import home
    from datetime import datetime, timezone

    projects_root = home() / "projects"
    if not projects_root.exists():
        console.print("[dim]No projects directory found.[/]")
        raise typer.Exit(0)

    cutoff = datetime.now(timezone.utc).timestamp() - (older_than * 86400)
    cleaned = 0
    freed_bytes = 0

    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue
        if project and project_dir.name != project:
            continue

        source_dir = project_dir / "source"
        if not source_dir.exists():
            continue

        for ch_dir in sorted(source_dir.iterdir()):
            if not ch_dir.is_dir():
                continue
            for img in ch_dir.iterdir():
                if img.is_file() and img.stat().st_mtime < cutoff:
                    freed_bytes += img.stat().st_size
                    img.unlink()
                    cleaned += 1

    console.print(f"[bold green]✓ Cleaned[/] {cleaned} files, freed {freed_bytes / (1024 * 1024):.1f} MB")


@app.callback()
def main(ctx: typer.Context):
    """Manga/manhwa translation pipeline."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_interactive())


# ── Interactive TUI ──────────────────────────────────────────────


async def _interactive():
    from rich.console import Console
    from ..cli.tui import TUI, load_projects
    from ..config import load_config as _load_cfg

    _, paths = _load_cfg()
    paths.ensure()

    log_file = paths.cache / "last_run.log"
    tui = TUI(log_file=log_file)
    projects, chapters_map = await load_projects()
    tui.set_projects(projects, chapters_map)
    tui.start()

    last_summary = None
    try:
        while True:
            result = await tui.browse()
            if result.action == "quit":
                break
            if result.action == "resume" and result.project:
                input_str = result.project.get("source_url") or result.project.get("title", "")
            else:
                input_str = result.input_value
            tui.switch_to_pipeline()
            last_summary = await run_pipeline(
                tui, input_str, result.force, result.from_ch, result.to_ch, paths)
            await tui.wait_for_key()
            projects, chapters_map = await load_projects()
            tui.set_projects(projects, chapters_map)
    finally:
        tui.stop()

    # Post-TUI: restore user context lost when alternate screen buffer
    # is released.
    if last_summary:
        console = Console()
        console.print(
            f"[bold green]✓ Done[/] [cyan]{last_summary['name']}[/] — "
            f"{last_summary['done']} done, {last_summary['skipped']} skipped, "
            f"{last_summary['failed']} failed"
        )
        console.print(f"[bold]Output:[/] [cyan]{last_summary['out_root']}[/]")
        console.print(f"[dim]Log:[/]    [cyan]{log_file}[/]")
