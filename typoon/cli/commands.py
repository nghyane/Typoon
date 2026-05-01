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
    input: Path = typer.Argument(..., help="Raw chapter image folder"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
    run_id: str = typer.Option("latest", "--run-id", help="Debug run id"),
    debug_root: Path | None = typer.Option(None, "--debug-root", help="Debug runs root"),
):
    """Prepare raw images into a chapter workspace."""
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
        console.print(f"  Output:    [cyan]{out_dir}[/]")
        console.print(f"  Debug run: [cyan]{artifacts.root}[/]")


@app.command()
def scan(
    input: Path = typer.Argument(..., help="Chapter workspace directory"),
    run_id: str = typer.Option("latest", "--run-id", help="Debug run id"),
    debug_root: Path | None = typer.Option(None, "--debug-root", help="Debug runs root"),
):
    """Run vision scan on a prepared chapter workspace."""
    from ..adapters.mask_store import MaskStore
    from ..adapters.vision_runtime import VisionRuntime
    from ..domain.prepared import Chapter
    from ..runs import FileArtifactSink
    from ..stages import scan_chapter

    if not (input / "manifest.json").exists():
        console.print(f"[red]Not a prepared chapter (missing manifest.json):[/] {input}")
        raise typer.Exit(2)

    chapter = Chapter.load(input)
    debug_dir = debug_root or input / "debug-runs"
    artifacts = FileArtifactSink(debug_dir, run_id)

    runtime, _, _ = VisionRuntime.from_config()
    result = scan_chapter(chapter, runtime, artifacts=artifacts)

    result.chapter.save(input)
    result.masks.save(input)

    total_bubbles = sum(len(p.bubbles) for p in result.chapter.pages)
    console.print(f"[green]✓ Scanned[/] {chapter.page_count} pages, {total_bubbles} bubbles")
    console.print(f"  Scan output: [cyan]{input}/scan/[/]")
    console.print(f"  Debug run:   [cyan]{artifacts.root}[/]")


@app.command()
def translate(
    input: Path = typer.Argument(..., help="Chapter workspace directory (must have scan/ output)"),
    project_id: int = typer.Option(1, "--project-id", "-p", help="Project ID in database"),
    chapter: float = typer.Option(1.0, "--chapter", "-c", help="Chapter number"),
    source_lang: str = typer.Option("ko", "--source-lang", "-s", help="Source language"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t", help="Target language"),
    db: Path | None = typer.Option(None, "--db", help="SQLite database path (default: ~/.typoon/typoon.db)"),
):
    """Translate a scanned chapter using LLM agents."""
    asyncio.run(_translate(input, project_id, chapter, source_lang, target_lang, db))


async def _translate(
    workspace: Path,
    project_id: int,
    chapter_num: float,
    source_lang: str,
    target_lang: str,
    db_path: Path | None,
):
    from ..adapters.session import make_session
    from ..domain.scan import Chapter as ScannedChapter
    from ..paths import home
    from ..stages import translate_chapter
    from ..storage.sqlite import SqliteStore

    scan_dir = workspace / "scan"
    if not (scan_dir / "manifest.json").exists():
        console.print(f"[red]No scan output found at {scan_dir}[/]")
        console.print("[dim]Run: typoon scan <workspace>[/]")
        raise typer.Exit(2)

    db_path = db_path or home() / "typoon.db"
    store = await SqliteStore.open(db_path)

    try:
        scanned = ScannedChapter.load(workspace)
        session = make_session(
            project_id=project_id,
            chapter=chapter_num,
            source_lang=source_lang,
            target_lang=target_lang,
            store=store,
        )

        console.print(f"[dim]Translating {scanned.page_count} pages, "
                      f"{len(scanned.all_bubbles)} bubbles…[/]")
        translated = await translate_chapter(scanned, session)
        translated.save(workspace)

        non_skip = sum(1 for b in translated.all_bubbles if b.kind != "skip")
        console.print(f"[green]✓ Translated[/] {non_skip}/{len(translated.all_bubbles)} bubbles")
        console.print(f"  Output: [cyan]{workspace}/translate/[/]")
    finally:
        await store.close()


@app.command()
def render(
    input: Path = typer.Argument(..., help="Chapter workspace directory (must have translate/ output)"),
):
    """Render translated chapter — erase source text, draw translations."""
    from ..adapters.mask_store import MaskStore
    from ..adapters.vision_runtime import VisionRuntime
    from ..domain.translate import Chapter as TranslatedChapter
    from ..stages import render_chapter

    translate_dir = input / "translate"
    if not (translate_dir / "manifest.json").exists():
        console.print(f"[red]No translate output found at {translate_dir}[/]")
        console.print("[dim]Run: typoon translate <workspace>[/]")
        raise typer.Exit(2)

    translated = TranslatedChapter.load(input)
    masks = MaskStore.load(input)
    runtime, _, _ = VisionRuntime.from_config()

    result = render_chapter(translated, masks, runtime, out_dir=input)

    overflows = sum(1 for b in result.pages for rb in b.bubbles if rb.overflow)
    console.print(f"[green]✓ Rendered[/] {result.page_count} pages")
    if overflows:
        console.print(f"  [yellow]⚠ {overflows} overflow(s)[/]")
    console.print(f"  Output: [cyan]{input}/render/[/]")


@app.command()
def run(
    input: Path = typer.Argument(..., help="Raw chapter image folder"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Workspace output directory"),
    project_id: int = typer.Option(1, "--project-id", "-p"),
    chapter: float = typer.Option(1.0, "--chapter", "-c"),
    source_lang: str = typer.Option("ko", "--source-lang", "-s"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    db: Path | None = typer.Option(None, "--db"),
):
    """Run full pipeline: prepare → scan → translate → render."""
    asyncio.run(_run(input, out, project_id, chapter, source_lang, target_lang, db))


async def _run(
    raw_dir: Path,
    out: Path | None,
    project_id: int,
    chapter_num: float,
    source_lang: str,
    target_lang: str,
    db_path: Path | None,
):
    from ..adapters.local_source import LocalSource
    from ..adapters.mask_store import MaskStore
    from ..adapters.session import make_session
    from ..adapters.vision_runtime import VisionRuntime
    from ..paths import home
    from ..stages import prepare_chapter, render_chapter, scan_chapter, translate_chapter
    from ..storage.sqlite import SqliteStore

    workspace = out or _default_prepared_dir(raw_dir)
    db_path = db_path or home() / "typoon.db"

    # Prepare
    console.print(f"[dim]Preparing {raw_dir.name}…[/]")
    chapter = prepare_chapter(LocalSource(raw_dir), workspace, source_label=str(raw_dir))
    console.print(f"  [green]✓[/] prepare  {chapter.page_count} pages")

    # Scan
    console.print("[dim]Scanning…[/]")
    runtime, _, _ = VisionRuntime.from_config()
    scan_out = scan_chapter(chapter, runtime)
    scan_out.chapter.save(workspace)
    scan_out.masks.save(workspace)
    total_bubbles = len(scan_out.chapter.all_bubbles)
    console.print(f"  [green]✓[/] scan     {total_bubbles} bubbles")

    # Translate
    console.print("[dim]Translating…[/]")
    store = await SqliteStore.open(db_path)
    try:
        session = make_session(
            project_id=project_id,
            chapter=chapter_num,
            source_lang=source_lang,
            target_lang=target_lang,
            store=store,
        )
        translated = await translate_chapter(scan_out.chapter, session)
        translated.save(workspace)
        non_skip = sum(1 for b in translated.all_bubbles if b.kind != "skip")
        console.print(f"  [green]✓[/] translate {non_skip}/{total_bubbles} bubbles")
    finally:
        await store.close()

    # Render
    console.print("[dim]Rendering…[/]")
    result = render_chapter(translated, scan_out.masks, runtime, out_dir=workspace)
    console.print(f"  [green]✓[/] render   {result.page_count} pages")

    console.print(f"\n[bold green]Done.[/] Output: [cyan]{workspace}/render/[/]")


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


# ── Helpers ──────────────────────────────────────────────────────────


def _prepare_inputs(path: Path) -> list[Path]:
    chapters = [child for child in sorted(path.iterdir()) if child.is_dir() and _has_images(child)]
    return chapters or [path]


def _default_prepared_dir(chapter_input: Path) -> Path:
    if chapter_input.parent.name == "source":
        return chapter_input.parent.parent / "prepared" / chapter_input.name
    return chapter_input / "prepared"


def _default_debug_root(chapter_input: Path) -> Path:
    if chapter_input.parent.name == "source":
        return chapter_input.parent.parent / "debug-runs"
    return chapter_input / "debug-runs"


def _is_project_chapter(chapter_input: Path) -> bool:
    return chapter_input.parent.name == "source"


def _has_images(path: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    return any(child.is_file() and child.suffix.lower() in exts for child in path.iterdir())
