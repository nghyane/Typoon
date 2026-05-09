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
def api(
    host:   str  = typer.Option(None, "--host", help="Bind host (default from [server].host)"),
    port:   int  = typer.Option(None, "--port", "-p", help="Bind port (default from [server].port)"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)"),
):
    """Run the HTTP API (FastAPI on uvicorn).

    DATABASE_URL must point at a Postgres instance. Workers can run in
    the same process via `typoon work --role full` or on a separate
    host (vision/llm) sharing the same DB.
    """
    import logging
    import uvicorn
    from ..config import load_config
    cfg, _ = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "typoon.api.app:app",
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
        # Long-lived SSE streams (/api/events) keep connections open
        # forever from uvicorn's perspective; without a graceful timeout
        # `Ctrl+C` hangs at "Waiting for connections to close" until the
        # last browser tab closes its EventSource.
        timeout_graceful_shutdown=5,
    )


@app.command()
def work(
    role:        str = typer.Option("full", "--role", "-r",
                                    help="vision | llm | api | storage | full"),
    concurrency: int = typer.Option(3, "--concurrency", "-c",
                                    help="Translate workers (only used when role=llm/full)"),
):
    """Start pipeline workers for a deployment role.

    full     everything in-process (default; dev on Mac)
    vision   scan + render only (GPU node)
    llm      translate only (LLM I/O node)
    api      no worker loops (API server only)
    storage  no worker loops; storage role lives in the API process
             (run `typoon api` with TYPOON_API_ROLE=storage)
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
        console.print(
            f"[red]invalid role: {role}[/] — use vision|llm|api|storage|full",
        )
        raise typer.Exit(1)
    console.print(f"[dim]workers started role={role_enum} translate×{concurrency} — Ctrl+C to stop[/]")
    await run_workers(role_enum, translate_concurrency=concurrency)


# ── pdf ───────────────────────────────────────────────────────────────


@app.command()
def export(
    slug:    str   = typer.Argument(..., help="Project slug"),
    formats: str   = typer.Option("pdf", "--format", "-f", help="Comma-separated: pdf,zip,webp"),
    range_:  str   = typer.Option("", "--range", "-r", help="Chapter range e.g. 1-3 or 5"),
    to:      Path  = typer.Option(None, "--to",   help="Override output dir (default: ~/.typoon/exports/<slug>/)"),
    force:   bool  = typer.Option(False, "--force", help="Re-export even if manifest is fresh"),
    open_:   bool  = typer.Option(True,  "--open/--no-open", help="Open the first PDF in system viewer"),
):
    """Export rendered chapters to a folder (PDF / zip / individual WebP)."""
    asyncio.run(_export(slug, formats, range_, to, force, open_))


async def _export(
    slug: str,
    formats: str,
    range_: str,
    to: Path | None,
    force: bool,
    open_: bool,
) -> None:
    from ..adapters.projects import Projects
    from ..adapters.storage_registry import build_storage
    from ..config import load_config
    from ..paths import Paths
    from ..stages.export import ChapterRef, export_chapters

    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    valid = {"pdf", "zip", "webp"}
    bad = [f for f in fmt_list if f not in valid]
    if bad:
        console.print(f"[red]Unknown format(s): {', '.join(bad)}. Choose from {sorted(valid)}.[/]")
        return

    lo, hi = _parse_range(range_)

    projects = await Projects.open()
    try:
        proj = await projects.require(slug)
        all_chs = await projects._db.get_all_chapters(proj["id"])
        if not all_chs:
            console.print("[yellow]No chapters in project.[/]")
            return

        # Filter to rendered chapters within the requested range.
        refs: list[ChapterRef] = []
        for ch in all_chs:
            if lo is not None and ch["idx"] < lo:
                continue
            if hi is not None and ch["idx"] > hi:
                continue
            if not ch.get("rendered") or not ch.get("archive_backend") or not ch.get("archive_locator"):
                continue
            refs.append(ChapterRef(
                chapter_id=ch["id"],
                chapter_idx=ch["idx"],
                archive_backend=ch["archive_backend"],
                archive_locator=ch["archive_locator"],
                rendered_at=str(ch.get("updated_at") or ""),
            ))

        if not refs:
            console.print("[yellow]No rendered chapters match the range.[/]")
            return

        config, paths = load_config()
        stores = build_storage(config, paths)
        dest = (to / slug) if to else (paths.exports / slug)

        results = await export_chapters(
            slug=slug,
            chapters=refs,
            formats=fmt_list,
            stores=stores,
            dest_dir=dest,
            force=force,
        )

        for r in results:
            for fmt, path in r.formats.items():
                console.print(f"[green]✓[/] ch{r.chapter_idx:.4g} [{fmt}]  {path}")

        if open_ and "pdf" in fmt_list and results:
            first_pdf = results[0].formats.get("pdf")
            if first_pdf:
                _open_in_viewer(first_pdf)
    finally:
        await projects.close()


def _parse_range(spec: str) -> tuple[float | None, float | None]:
    spec = spec.strip()
    if not spec:
        return None, None
    if "-" in spec:
        lo_s, hi_s = (s.strip() for s in spec.split("-", 1))
        return (float(lo_s) if lo_s else None, float(hi_s) if hi_s else None)
    n = float(spec)
    return n, n


def _open_in_viewer(path: Path) -> None:
    import subprocess
    import sys
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(path)])
    else:
        subprocess.Popen(["start", str(path)], shell=True)


# ── debug-scan ────────────────────────────────────────────────────────


@app.command("debug-scan")
def debug_scan(
    project_id: int = typer.Argument(..., help="Project id (DB row id)"),
    chapter_id: int = typer.Argument(..., help="Chapter id (DB row id)"),
    pages:      str = typer.Option("", "--pages", "-p",
                                   help="Comma-separated page indices (default: all)"),
    out:        Path = typer.Option(Path("debug-runs"), "--out", "-o",
                                    help="Output root for run artifacts"),
):
    """Re-run scan on a prepared chapter and write grouping overlays.

    Reads the existing prepared.bnl from the artifact store (no re-prepare),
    runs the full vision pipeline (detect → group → OCR → filter), and writes:

    \b
      <out>/p<P>_c<C>/02_detect/page_*_detect.png      bounding boxes
      <out>/p<P>_c<C>/03_group/page_*_groups.png       accepted groups
      <out>/p<P>_c<C>/03_group/page_*_inspect.png      4-panel: units|groups|masks|erased
      <out>/p<P>_c<C>/02_detect/page_*_state.json      raw ScanState dump
    """
    indices = [int(s) for s in pages.split(",") if s.strip()] if pages else None
    asyncio.run(_debug_scan(project_id, chapter_id, indices, out))


async def _debug_scan(
    project_id: int,
    chapter_id: int,
    page_indices: list[int] | None,
    out_root: Path,
) -> None:
    import tempfile

    from ..adapters.storage_registry import build_storage
    from ..adapters.chapter_archive import prepared_key
    from ..adapters.loader import open_prepared_reader
    from ..adapters.vision_runtime import VisionRuntime
    from ..config import load_config
    from ..paths import Paths
    from ..runs.artifacts import FileArtifactSink
    from ..vision.inspect import state_to_dict, write_inspection

    config, paths = load_config()
    paths.ensure()

    # debug-scan reads prepared.bnl which is always on the local store.
    store    = build_storage(config, paths).pipeline
    runtime, *_ = VisionRuntime.from_config(config, paths)
    run_id   = f"p{project_id}_c{chapter_id}"
    sink     = FileArtifactSink(out_root, run_id, clean=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        try:
            reader = await open_prepared_reader(
                store, prepared_key(project_id, chapter_id), tmp,
            )
        except FileNotFoundError as e:
            console.print(f"[red]prepared archive not found:[/] {e}")
            raise typer.Exit(1)

        with reader:
            total = reader.page_count
            targets = page_indices if page_indices is not None else list(range(total))
            invalid = [i for i in targets if i < 0 or i >= total]
            if invalid:
                console.print(f"[red]invalid page indices:[/] {invalid} (chapter has {total} pages)")
                raise typer.Exit(2)

            for page_index in targets:
                console.print(f"  scan page {page_index + 1}/{total}…")
                image = reader.read_rgb(page_index)
                state = runtime.scan_page_state(image)

                # Per-stage overlays — same writers used by scan_chapter().
                from ..stages.scan import _write_artifacts
                _write_artifacts(sink, page_index, image, state)

                # 4-panel inspect: text_boxes | groups | masks | erased.
                # `write_inspection` writes one PNG per page; route it under 03_group.
                write_inspection(
                    sink.root / "03_group",
                    page_index,
                    image,
                    state,
                    runtime.eraser,
                )

                # Full state JSON for offline diagnosis (units, scopes, groups).
                sink.write_json(
                    "02_detect", f"page_{page_index:04d}_full_state.json",
                    state_to_dict(page_index, "", state),
                )

    console.print(f"\n[green]✓[/] artifacts → {sink.root}")
    console.print(f"  open: [dim]{sink.root}/03_group[/]")


# ── prune ─────────────────────────────────────────────────────────────


@app.command()
def prune(
    days:        int  = typer.Option(30, "--days", "-d",
                                     help="Prune cache for chapters not updated in N days"),
    events_days: int  = typer.Option(7, "--events-days",
                                     help="Drop SSE events older than N days"),
    dry_run:     bool = typer.Option(False, "--dry-run",
                                     help="Show what would be deleted, don't delete"),
):
    """Delete intermediate caches and trim the SSE event table.

    - prepared.bnl + masks.npz for chapters not updated in --days.
      Render archives are NOT touched — readers keep working.
    - events table rows older than --events-days. Clients past that
      lose their SSE replay buffer; the live stream keeps working.
    """
    asyncio.run(_prune(days, events_days, dry_run))


async def _prune(days: int, events_days: int, dry_run: bool) -> None:
    from ..adapters.chapter_archive import masks_key, prepared_key
    from ..adapters.storage_registry import build_storage
    from ..config import load_config
    from ..storage import PostgresStore

    config, paths = load_config()
    paths.ensure()
    db = await PostgresStore.open(config.database_url)
    try:
        rows = await db.list_prunable_chapters(days)
        if rows:
            local = build_storage(config, paths).pipeline
            prepared_freed = 0
            masks_freed = 0
            prepared_count = 0
            masks_count = 0
            for r in rows:
                pid, cid = r["project_id"], r["chapter_id"]
                for key, label in (
                    (prepared_key(pid, cid), "prepared"),
                    (masks_key(pid, cid), "masks"),
                ):
                    path = paths.artifacts / key
                    if not path.exists():
                        continue
                    size = path.stat().st_size
                    if dry_run:
                        console.print(f"[dim]would delete[/] {key} ({_human(size)})")
                    else:
                        await local.delete(key)
                    if label == "prepared":
                        prepared_count += 1
                        prepared_freed += size
                    else:
                        masks_count += 1
                        masks_freed += size

            verb = "would free" if dry_run else "freed"
            console.print(
                f"[green]✓[/] {verb} "
                f"[bold]{_human(prepared_freed + masks_freed)}[/] "
                f"({prepared_count} prepared, {masks_count} masks "
                f"across {len(rows)} chapter{'s' if len(rows) != 1 else ''})",
            )
        else:
            console.print(f"[dim]No chapters older than {days}d to prune.[/]")

        if dry_run:
            async with db._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS n FROM events "
                    "WHERE created_at < NOW() - make_interval(days => $1)",
                    events_days,
                )
            console.print(f"[dim]would drop {row['n']} event row(s) older than {events_days}d[/]")
        else:
            n = await db.prune_events(events_days)
            console.print(f"[green]✓[/] dropped {n} event row(s) older than {events_days}d")
    finally:
        await db.close()


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


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
