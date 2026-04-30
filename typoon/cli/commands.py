"""CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from ..adapters.connectors import get_connectors

app = typer.Typer(name="typoon")
console = Console()


@app.command()
def status():
    """Show connectors and authentication status."""
    console.print("[bold]Connectors[/]\n")
    for c in get_connectors():
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
    connector = next((c for c in get_connectors() if site in c.site_name or c.site_name in site), None)
    if connector is None:
        console.print(f"[red]Unknown site: {site}[/]")
        raise typer.Exit(1)
    console.print(f"[yellow]Solving Cloudflare challenge for {connector.site_name}…[/]")
    console.print("[dim]Complete the challenge in the browser window.[/]")
    await connector.authenticate()
    console.print(f"[green]✓[/] {connector.site_name} — authenticated")


@app.command()
def prepare(
    input: Path = typer.Argument(..., help="Raw chapter folder or project source folder"),
    out: Path | None = typer.Option(None, "--out", "-o", help="PreparedChapter output directory"),
    run_id: str = typer.Option("latest", "--run-id", help="Debug run id"),
    debug_root: Path | None = typer.Option(None, "--debug-root", help="Debug runs root"),
):
    """Prepare raw images into PreparedChapter output."""
    from ..adapters.local_source import LocalSource
    from ..runs import FileArtifactSink
    from ..stages import prepare_chapter

    if not input.is_dir():
        console.print(f"[red]Input must be an image folder:[/] {input}")
        raise typer.Exit(2)

    rid = run_id or uuid4().hex[:12]
    chapters = _prepare_inputs(input)
    if len(chapters) > 1 and out is not None:
        console.print("[red]--out is only valid when preparing a single chapter folder.[/]")
        raise typer.Exit(2)

    for chapter_input in chapters:
        out_dir = out or _default_prepared_dir(chapter_input)
        debug_dir = debug_root or _default_debug_root(chapter_input)
        run_name = f"{rid}/{chapter_input.name}" if _is_project_chapter(chapter_input) else rid
        artifacts = FileArtifactSink(debug_dir, run_name)
        chapter = prepare_chapter(
            LocalSource(chapter_input),
            out_dir,
            source_label=str(chapter_input),
            artifacts=artifacts,
        )
        console.print(f"[green]✓ Prepared[/] {chapter_input.name}  {chapter.page_count} pages")
        console.print(f"  PreparedChapter: [cyan]{out_dir}[/]")
        console.print(f"  Debug run:       [cyan]{artifacts.root}[/]")


def _prepare_inputs(path: Path) -> list[Path]:
    chapters = [child for child in sorted(path.iterdir()) if child.is_dir() and _has_images(child)]
    return chapters or [path]


def _default_prepared_dir(chapter_input: Path) -> Path:
    if chapter_input.parent.name == "source":
        return chapter_input.parent.parent / "prepared" / chapter_input.name
    return chapter_input / "PreparedChapter"


def _default_debug_root(chapter_input: Path) -> Path:
    if chapter_input.parent.name == "source":
        return chapter_input.parent.parent / "debug-runs"
    return chapter_input / "debug-runs"


def _is_project_chapter(chapter_input: Path) -> bool:
    return chapter_input.parent.name == "source"


def _has_images(path: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    return any(child.is_file() and child.suffix.lower() in exts for child in path.iterdir())


@app.command()
def scan(
    input: Path = typer.Argument(..., help="PreparedChapter directory (contains manifest.json)"),
    run_id: str = typer.Option("latest", "--run-id", help="Debug run id"),
    debug_root: Path | None = typer.Option(None, "--debug-root", help="Debug runs root"),
):
    """Run vision scan on a PreparedChapter and write debug artifacts."""
    from ..adapters.vision_runtime import VisionRuntime
    from ..domain.prepared import load_prepared_chapter
    from ..runs import FileArtifactSink
    from ..stages import scan_chapter

    if not (input / "manifest.json").exists():
        console.print(f"[red]Not a PreparedChapter (missing manifest.json):[/] {input}")
        raise typer.Exit(2)

    chapter = load_prepared_chapter(input)
    debug_dir = debug_root or input / "debug-runs"
    artifacts = FileArtifactSink(debug_dir, run_id)

    runtime, _, _ = VisionRuntime.from_config()
    result = scan_chapter(chapter, runtime, artifacts=artifacts)

    total_bubbles = sum(len(p.bubbles) for p in result.chapter.pages)
    console.print(f"[green]✓ Scanned[/] {chapter.page_count} pages, {total_bubbles} bubbles")
    console.print(f"  Debug run: [cyan]{artifacts.root}[/]")


@app.command()
def translate(
    input: str = typer.Argument(..., help="URL or local path"),
    source_lang: str = typer.Option("", "--source-lang", "-s"),
    target_lang: str = typer.Option("", "--target-lang", "-t"),
    from_ch: float = typer.Option(0, "--from"),
    to_ch: float = typer.Option(0, "--to"),
    force: bool = typer.Option(False, "--force"),
):
    """Removed pending PreparedChapter pipeline."""
    console.print("[red]translate is unavailable:[/] legacy project pipeline was removed.")
    console.print("[dim]Build prepare -> PreparedChapter -> page-local pipeline next.[/]")
    raise typer.Exit(2)


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
