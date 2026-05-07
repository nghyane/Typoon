"""CLI — user interaction only."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..runs.events import Hook
from .events import render as render_event

app     = typer.Typer(name="typoon", help="Manga translation pipeline.")
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
    connector = next(
        (c for c in get_connectors() if site in c.site_name or c.site_name in site), None
    )
    if connector is None:
        console.print(f"[red]Unknown site:[/] {site}")
        raise typer.Exit(1)
    console.print(f"[yellow]Opening browser for {connector.site_name}…[/]")
    await connector.authenticate()
    console.print(f"[green]✓[/] {connector.site_name} — authenticated")


# ── pull ──────────────────────────────────────────────────────────────


@app.command()
def pull(
    target:      str   = typer.Argument(..., help="URL (new project) or slug (existing)"),
    url:         str   = typer.Argument(None, help="Manga URL when slug given first"),
    target_lang: str   = typer.Option("vi", "--target-lang", "-t"),
    from_ch:     float = typer.Option(0, "--from", help="First chapter (0 = ask)"),
    to_ch:       float = typer.Option(0, "--to",   help="Last chapter"),
):
    """Download chapters from URL and enqueue for translation.

    \b
    Examples:
      typoon pull https://comix.to/manga/solo-leveling/ -t vi
      typoon pull solo-leveling https://comix.to/manga/solo-leveling/ --from 10
    """
    asyncio.run(_pull(target, url, target_lang, from_ch, to_ch))


async def _pull(
    target: str,
    url: str | None,
    target_lang: str,
    from_ch: float,
    to_ch: float,
) -> None:
    from ..adapters.projects import Projects
    projects = await Projects.open()
    try:
        if url is not None:
            await projects.require(target)
            info     = await projects.discover(url)
            selected = _select_chapters(info.chapters, from_ch, to_ch)
            if not selected:
                return
            console.print(f"\n[bold]{info.suggested_title}[/]  {len(selected)} chapter(s)\n")
            await projects.pull_more(target, url, info, selected, ConsoleHook())
        else:
            info     = await projects.discover(target)
            selected = _select_chapters(info.chapters, from_ch, to_ch)
            if not selected:
                return
            console.print(f"\n[bold]{info.suggested_title}[/]  {len(selected)} chapter(s)\n")
            slug = await projects.pull_new(info, target, selected, target_lang, ConsoleHook())
            console.print(f"\n[dim]slug: [bold]{slug}[/][/]")
    finally:
        await projects.close()


# ── add ───────────────────────────────────────────────────────────────


@app.command()
def add(
    target:      str  = typer.Argument(..., help="Folder path or project slug"),
    folder:      Path = typer.Argument(None, help="Chapter folder (when slug given first)"),
    target_lang: str  = typer.Option("vi", "--target-lang", "-t"),
    source_lang: str  = typer.Option("ko", "--source-lang", "-s"),
    name:        str  = typer.Option(None, "--name", "-n"),
):
    """Import local chapter images and enqueue for translation.

    \b
    Examples:
      typoon add ./solo-leveling/ -s ko -t vi
      typoon add solo-leveling ./ch015/
    """
    asyncio.run(_add(target, folder, target_lang, source_lang, name))


async def _add(
    target: str,
    folder: Path | None,
    target_lang: str,
    source_lang: str,
    name: str | None,
) -> None:
    from ..adapters.projects import Projects
    projects = await Projects.open()
    try:
        if folder is not None:
            await projects.require(target)
            src = Path(folder)
            if not src.is_dir():
                console.print(f"[red]Not a directory:[/] {src}")
                raise typer.Exit(2)
            await projects.import_more(target, src, ConsoleHook())
        else:
            src = Path(target)
            if not src.is_dir():
                console.print(f"[red]Not a directory:[/] {src}")
                raise typer.Exit(2)
            project_id = await projects.import_new(src, name or src.name, source_lang, target_lang, ConsoleHook())
            proj = await projects._db.get_project(project_id)
            slug = proj["slug"]
            console.print(f"\n[dim]slug: [bold]{slug}[/][/]")
    finally:
        await projects.close()


# ── redo ──────────────────────────────────────────────────────────────


@app.command()
def redo(
    slug:    str   = typer.Argument(..., help="Project slug"),
    from_ch: float = typer.Option(0, "--from", help="First chapter"),
    to_ch:   float = typer.Option(0, "--to",   help="Last chapter"),
):
    """Reset chapters and re-run the full pipeline from scratch.

    Deletes all artifacts and derived data, then re-enqueues scan.
    Source images (pages/) are kept.

    \b
    Examples:
      typoon redo solo-leveling
      typoon redo solo-leveling --from 5 --to 10
    """
    asyncio.run(_redo(slug, from_ch, to_ch))


async def _redo(slug: str, from_ch: float, to_ch: float) -> None:
    from ..adapters.projects import Projects
    projects = await Projects.open()
    try:
        proj    = await projects.require(slug)
        indices: list[float] | None = None
        if from_ch > 0 or to_ch > 0:
            all_chs = await projects._db.get_all_chapters(proj["id"])
            lo      = from_ch or all_chs[0]["idx"]
            hi      = to_ch   or all_chs[-1]["idx"]
            indices = [c["idx"] for c in all_chs if lo <= c["idx"] <= hi]
        count = await projects.redo(slug, indices)
        console.print(f"[green]✓[/] {count} chapter(s) reset and enqueued — run 'typoon work' to process")
    finally:
        await projects.close()


# ── work ──────────────────────────────────────────────────────────────


@app.command()
def work(
    role:        str = typer.Option("full", "--role", "-r",
                                    help="vision | llm | api | full"),
    concurrency: int = typer.Option(3, "--concurrency", "-c",
                                    help="Translate workers (only used when role=llm/full)"),
):
    """Start pipeline workers for a deployment role.

    full    everything in-process (default; SQLite-compatible)
    vision  scan + render only (GPU node)
    llm     translate only (LLM I/O node)
    api     no worker loops (API server only)
    """
    asyncio.run(_work(role, concurrency))


async def _work(role: str, concurrency: int) -> None:
    import logging
    from ..workers.loop import Role, run_workers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        role_enum = Role(role)
    except ValueError:
        console.print(f"[red]invalid role: {role}[/] — use vision|llm|api|full")
        raise typer.Exit(1)
    console.print(f"[dim]workers started role={role_enum} translate×{concurrency} — Ctrl+C to stop[/]")
    await run_workers(role_enum, translate_concurrency=concurrency)


# ── pdf ───────────────────────────────────────────────────────────────


@app.command()
def pdf(
    slug:    str   = typer.Argument(..., help="Project slug"),
    from_ch: float = typer.Option(0, "--from", help="First chapter"),
    to_ch:   float = typer.Option(0, "--to",   help="Last chapter"),
    out:     Path  = typer.Option(None, "--out", "-o", help="Output path (default: <slug>-ch<N>.pdf)"),
):
    """Export rendered chapter(s) to PDF and open in system viewer."""
    asyncio.run(_pdf(slug, from_ch, to_ch, out))


async def _pdf(slug: str, from_ch: float, to_ch: float, out: Path | None) -> None:
    import io
    import subprocess
    import sys
    import tempfile

    import bunle
    from PIL import Image

    from ..adapters.artifact_store import LocalArtifactStore
    from ..adapters.chapter_archive import render_key
    from ..adapters.projects import Projects
    from ..paths import Paths

    projects = await Projects.open()
    try:
        proj     = await projects.require(slug)
        all_chs  = await projects._db.get_all_chapters(proj["id"])
        lo       = from_ch or (all_chs[0]["idx"] if all_chs else 1)
        hi       = to_ch   or lo
        chapters = [c for c in all_chs if lo <= c["idx"] <= hi]
        if not chapters:
            console.print("[red]No chapters found in range.[/]")
            return
        paths = Paths()
        store = LocalArtifactStore(paths.artifacts)
        for ch in chapters:
            state = await projects._db.get_chapter_render_state(ch["id"])
            if state is None or not state["rendered"]:
                console.print(f"[yellow]ch{ch['idx']:.4g}: not rendered, skipping[/]")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "render.bnl"
                await store.get_file(render_key(proj["id"], ch["id"]), local)
                with bunle.Reader(str(local)) as reader:
                    images = [
                        Image.open(io.BytesIO(reader.page(i))).convert("RGB")
                        for i in range(reader.page_count)
                    ]
            if not images:
                console.print(f"[yellow]ch{ch['idx']:.4g}: empty render archive, skipping[/]")
                continue
            dest = out or Path(f"{slug}-ch{ch['idx']:.4g}.pdf")
            images[0].save(dest, save_all=True, append_images=images[1:])
            console.print(f"[green]✓[/] {dest} ({len(images)} pages)")
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(dest)])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(dest)])
            else:
                subprocess.Popen(["start", str(dest)], shell=True)
    finally:
        await projects.close()


# ── status ────────────────────────────────────────────────────────────


@app.command()
def status(slug: str = typer.Argument(None, help="Project slug (omit for all)")):
    """Show project and chapter progress."""
    asyncio.run(_status(slug))


async def _status(slug: str | None) -> None:
    from ..adapters.projects import Projects
    projects = await Projects.open()
    try:
        all_status = await projects.get_status(slug)
        if not all_status:
            msg = f"Project '{slug}' not found." if slug else "No projects. Run: typoon add <folder>"
            console.print(f"[dim]{msg}[/]")
            return
        for proj in all_status:
            _print_project(proj)
    finally:
        await projects.close()


def _print_project(proj) -> None:
    done  = sum(1 for c in proj.chapters if c.state == "done")
    total = len(proj.chapters)
    console.print(
        f"\n[bold]{proj.title}[/]  [dim]{proj.slug}[/]  "
        f"{proj.source_lang}→{proj.target_lang}  "
        f"[green]{done}[/]/[dim]{total}[/] done"
    )
    if not proj.chapters:
        console.print("  [dim]No chapters[/]")
        return
    t = Table(show_header=False, box=None, padding=(0, 1))
    for ch in proj.chapters:
        icon = {
            "done":    "[green]✓[/]",
            "running": "[yellow]⟳[/]",
            "error":   "[red]✗[/]",
            "pending": "[dim]○[/]",
            "idle":    "[dim]–[/]",
        }.get(ch.state, "?")
        detail = ""
        if ch.state == "done":
            detail = f"[dim]{ch.render_count} pages[/]"
        elif ch.state in ("running", "pending"):
            detail = f"[dim]{ch.stage}[/]"
        elif ch.state == "error":
            detail = f"[red]{ch.stage}: {ch.error[:60]}[/]"
        t.add_row(icon, f"[dim]#{ch.chapter_id}[/]", f"ch{ch.idx:.4g}", detail)
    console.print(t)


# ── helpers ───────────────────────────────────────────────────────────


def _select_chapters(chapters, from_ch: float, to_ch: float) -> list:
    if from_ch > 0 or to_ch > 0:
        lo = from_ch or chapters[0].number
        hi = to_ch   or chapters[-1].number
        return [c for c in chapters if lo <= c.number <= hi]
    console.print(f"  Available: ch{chapters[0].number:.0f}–ch{chapters[-1].number:.0f}")
    raw = typer.prompt("  Select chapters (e.g. 1-5 or 3)").strip()
    if "-" in raw:
        lo, hi = (float(x.strip()) for x in raw.split("-", 1))
    else:
        lo = hi = float(raw)
    return [c for c in chapters if lo <= c.number <= hi]
