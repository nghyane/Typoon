"""CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import typer
from rich.console import Console

from .output import SingleFileSource, save_pages
from .pipeline import translate_cli
from .resolve import _get_connectors

app = typer.Typer(name="typoon")
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
    from ..adapters.local_source import LocalSource
    from ..engine import Engine

    engine, _, paths = Engine.from_config()
    paths.output.mkdir(exist_ok=True)

    source = LocalSource(path) if path.is_dir() else SingleFileSource(path)
    pages, images = engine.preprocess(source)
    engine.erase(pages, images)

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
    from datetime import datetime, timezone

    from ..paths import home

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
