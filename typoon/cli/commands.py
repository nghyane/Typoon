"""CLI commands."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="typoon", help="Manga translation pipeline.")
console = Console()


# ── auth ──────────────────────────────────────────────────────────────


@app.command()
def auth(site: str = typer.Argument(..., help="Site to authenticate (e.g. comix.to)")):
    """Authenticate with a manga source site."""
    asyncio.run(_auth(site))


async def _auth(site: str):
    from ..adapters.connectors import get_connectors
    connector = next((c for c in get_connectors() if site in c.site_name or c.site_name in site), None)
    if connector is None:
        console.print(f"[red]Unknown site: {site}[/]")
        raise typer.Exit(1)
    console.print(f"[yellow]Opening browser for {connector.site_name}…[/]")
    await connector.authenticate()
    console.print(f"[green]✓[/] {connector.site_name} — authenticated")


# ── pull ──────────────────────────────────────────────────────────────


@app.command()
def pull(
    url: str = typer.Argument(..., help="Manga or chapter URL"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t", help="Target language"),
    from_ch: float = typer.Option(0, "--from", help="First chapter (0 = ask)"),
    to_ch: float = typer.Option(0, "--to", help="Last chapter (0 = ask)"),
    redo: str = typer.Option(None, "--redo", help="Re-run from stage: scan|translate|render"),
):
    """Download chapters from a manga URL and translate."""
    asyncio.run(_pull(url, target_lang, from_ch, to_ch, redo))


async def _pull(url: str, target_lang: str, from_ch: float, to_ch: float, redo: str | None):
    from ..adapters.connectors import get_connectors
    from ..downloader import download_images
    from ..paths import Paths, ProjectPaths, slugify
    from ..storage.sqlite import SqliteStore

    connector = next((c for c in get_connectors() if c.accepts(url)), None)
    if connector is None:
        console.print(f"[red]No connector for:[/] {url}")
        raise typer.Exit(1)

    console.print("[dim]Fetching chapter list…[/]")
    info = await connector.discover(url)
    console.print(f"  {info.suggested_title} — {len(info.chapters)} chapters ({info.suggested_lang})")

    # Select chapters
    chapters = _select_chapters(info.chapters, from_ch, to_ch)
    if not chapters:
        console.print("[yellow]No chapters selected.[/]")
        raise typer.Exit(0)

    paths = Paths()
    paths.ensure()
    slug = slugify(info.suggested_title, url)
    proj_paths = ProjectPaths(paths.projects, slug)
    proj_paths.ensure()

    db = await SqliteStore.open(paths.db)
    try:
        project_id = await db.get_or_create_project(
            title=info.suggested_title,
            source_lang=info.suggested_lang,
            target_lang=target_lang,
            source_url=url,
        )

        for ch in chapters:
            cp = proj_paths.chapter(ch.number)
            cp.ensure()

            if not any(cp.pages.iterdir()) if cp.pages.exists() else True:
                console.print(f"  [dim]Downloading ch{ch.number:03.0f}…[/]", end="")
                page_urls = await connector.get_page_urls(ch)
                headers = await connector._get_headers() if hasattr(connector, "_get_headers") else {}
                await download_images(page_urls, cp.pages, headers=headers)
                console.print(f" {len(page_urls)} pages")
            else:
                console.print(f"  ch{ch.number:03.0f}: images already present, skipping download")

            await db.add_chapter(project_id, ch.number, source_url=ch.best_variant.url)

        await _translate_chapters(db, project_id, [c.number for c in chapters], proj_paths, redo)
    finally:
        await db.close()


def _select_chapters(chapters, from_ch, to_ch):
    if from_ch > 0 or to_ch > 0:
        lo = from_ch or chapters[0].number
        hi = to_ch or chapters[-1].number
        return [c for c in chapters if lo <= c.number <= hi]

    # Interactive selection
    console.print(f"  First: ch{chapters[0].number:.0f}  Last: ch{chapters[-1].number:.0f}")
    raw = console.input("  Select chapters (e.g. 1-5 or 3): ").strip()
    if "-" in raw:
        parts = raw.split("-", 1)
        lo, hi = float(parts[0].strip()), float(parts[1].strip())
    else:
        lo = hi = float(raw)
    return [c for c in chapters if lo <= c.number <= hi]


# ── add ───────────────────────────────────────────────────────────────


@app.command()
def add(
    folder: Path = typer.Argument(..., help="Local folder with chapter images"),
    target_lang: str = typer.Option("vi", "--target-lang", "-t"),
    source_lang: str = typer.Option("ko", "--source-lang", "-s"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (default: folder name)"),
    redo: str = typer.Option(None, "--redo", help="Re-run from stage: scan|translate|render"),
):
    """Import a local folder of chapter images and translate."""
    asyncio.run(_add(folder, target_lang, source_lang, project, redo))


async def _add(folder: Path, target_lang: str, source_lang: str, project_name: str | None, redo: str | None):
    from ..paths import Paths, ProjectPaths, slugify
    from ..storage.sqlite import SqliteStore

    if not folder.is_dir():
        console.print(f"[red]Not a directory:[/] {folder}")
        raise typer.Exit(2)

    # Detect: single chapter or multi-chapter folder
    chapter_dirs = _detect_chapters(folder)
    console.print(f"  Detected {len(chapter_dirs)} chapter(s)")

    title = project_name or folder.name
    paths = Paths()
    paths.ensure()
    slug = slugify(title)
    proj_paths = ProjectPaths(paths.projects, slug)
    proj_paths.ensure()

    db = await SqliteStore.open(paths.db)
    try:
        project_id = await db.get_or_create_project(
            title=title,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        chapter_indices = []
        for i, src_dir in enumerate(chapter_dirs, start=1):
            # Try to parse chapter number from folder name, fallback to index
            ch_num = _parse_ch_num(src_dir.name) or float(i)
            cp = proj_paths.chapter(ch_num)
            cp.ensure()

            if not (cp.pages.exists() and any(cp.pages.iterdir())):
                console.print(f"  Copying ch{ch_num:03.0f} from {src_dir.name}…")
                _copy_images(src_dir, cp.pages)
            else:
                console.print(f"  ch{ch_num:03.0f}: already imported")

            await db.add_chapter(project_id, ch_num)
            chapter_indices.append(ch_num)

        await _translate_chapters(db, project_id, chapter_indices, proj_paths, redo)
    finally:
        await db.close()


def _detect_chapters(folder: Path) -> list[Path]:
    """Return list of image directories. Single chapter or multi-chapter."""
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    # Images directly in folder → single chapter
    if any(f.suffix.lower() in _IMAGE_EXTS for f in folder.iterdir() if f.is_file()):
        return [folder]
    # Subfolders containing images → multiple chapters
    subs = sorted(
        d for d in folder.iterdir()
        if d.is_dir() and any(f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir() if f.is_file())
    )
    return subs or [folder]


def _parse_ch_num(name: str) -> float | None:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def _copy_images(src: Path, dest: Path) -> None:
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
    for i, f in enumerate(files):
        shutil.copy2(f, dest / f"{i + 1:03d}{f.suffix.lower()}")


# ── translate ─────────────────────────────────────────────────────────


@app.command()
def translate(
    project: str = typer.Argument(None, help="Project name (omit for all pending)"),
    chapter: float = typer.Option(None, "--chapter", "-c", help="Specific chapter number"),
    redo: str = typer.Option(None, "--redo", help="Re-run from stage: scan|translate|render"),
):
    """Translate pending chapters (scan + translate + render)."""
    asyncio.run(_translate_cmd(project, chapter, redo))


async def _translate_cmd(project_name: str | None, chapter_num: float | None, redo: str | None):
    from ..paths import Paths, ProjectPaths
    from ..storage.sqlite import SqliteStore

    paths = Paths()
    db = await SqliteStore.open(paths.db)
    try:
        if project_name:
            proj = await db.get_project_by_title(project_name)
            if not proj:
                console.print(f"[red]Project not found:[/] {project_name}")
                raise typer.Exit(1)
            projects = [proj]
        else:
            projects = await db.list_projects()

        for proj in projects:
            proj_paths = ProjectPaths(paths.projects, _slug_for(proj))
            if chapter_num is not None:
                indices = [chapter_num]
            else:
                pending = await db.get_pending_chapters(proj["id"])
                indices = [c["idx"] for c in pending]

            if not indices:
                continue

            await _translate_chapters(db, proj["id"], indices, proj_paths, redo)
    finally:
        await db.close()


async def _translate_chapters(
    db,
    project_id: int,
    indices: list[float],
    proj_paths,
    redo: str | None,
):
    from ..adapters.session import make_session
    from ..adapters.vision_runtime import VisionRuntime
    from ..paths import Paths
    from ..stages.pipeline import run_chapter

    proj = await db.get_project(project_id)
    runtime, config, _ = VisionRuntime.from_config()

    for idx in indices:
        cp = proj_paths.chapter(idx)
        cp.ensure()
        console.print(f"\n[bold]ch{idx:03.0f}[/] {proj['title']}")

        await db.set_chapter_status(project_id, idx, "translating")
        try:
            session = make_session(
                project_id=project_id,
                chapter=idx,
                source_lang=proj["source_lang"],
                target_lang=proj["target_lang"],
                store=db,
                config=config,
            )
            await run_chapter(cp, session, runtime, redo_from=redo)
            await db.set_chapter_status(project_id, idx, "done")
            console.print(f"  [green]✓[/] done → {cp.render}")
        except Exception as e:
            await db.set_chapter_status(project_id, idx, "error")
            console.print(f"  [red]✗[/] {e}")


# ── status ────────────────────────────────────────────────────────────


@app.command()
def status(
    project: str = typer.Argument(None, help="Project name (omit for all)"),
):
    """Show project and chapter status."""
    asyncio.run(_status(project))


async def _status(project_name: str | None):
    from ..paths import Paths, ProjectPaths
    from ..storage.sqlite import SqliteStore

    paths = Paths()
    db = await SqliteStore.open(paths.db)
    try:
        projects = await db.list_projects()
        if not projects:
            console.print("[dim]No projects yet. Run: typoon pull <url>[/]")
            return

        for proj in projects:
            if project_name and proj["title"] != project_name:
                continue

            console.print(f"\n[bold]{proj['title']}[/] ({proj['source_lang']} → {proj['target_lang']})")
            chapters = await db.get_all_chapters(proj["id"])
            if not chapters:
                console.print("  [dim]No chapters imported[/]")
                continue

            proj_paths = ProjectPaths(paths.projects, _slug_for(proj))
            t = Table(show_header=False, box=None, padding=(0, 1))
            for ch in chapters:
                idx = ch["idx"]
                cp = proj_paths.chapter(idx)
                status_icon = {
                    "done":        "[green]✓[/]",
                    "translating": "[yellow]⟳[/]",
                    "error":       "[red]✗[/]",
                    "pending":     "[dim]○[/]",
                }.get(ch["status"], "?")
                render_info = f"{len(list(cp.render.iterdir()))} pages" if cp.is_rendered else ""
                t.add_row(
                    status_icon,
                    f"ch{idx:03.0f}",
                    ch["status"],
                    f"[dim]{render_info}[/]",
                )
            console.print(t)
    finally:
        await db.close()


# ── helpers ───────────────────────────────────────────────────────────


def _slug_for(proj: dict) -> str:
    from ..paths import slugify
    return slugify(proj["title"], proj.get("source_url") or "")
