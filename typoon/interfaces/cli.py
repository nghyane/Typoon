"""CLI entry point — thin wrapper around AppService.

Commands: typoon | typoon translate | typoon detect | typoon auth | typoon status
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import cv2
import typer
from rich.console import Console

app = typer.Typer(name="typoon", invoke_without_command=True)
console = Console()

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _has_images(path: Path) -> bool:
    return any(f.suffix.lower() in _IMAGE_EXTS for f in path.iterdir() if f.is_file())


def _has_chapter_subdirs(path: Path) -> bool:
    return any(d.is_dir() and _has_images(d) for d in path.iterdir())


def _parse_chapter_num(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0


def _discover_chapters(path: Path) -> list[Path]:
    dirs = [d for d in path.iterdir() if d.is_dir() and _has_images(d)]
    return sorted(dirs, key=lambda d: _parse_chapter_num(d.name))


def _ch_label(ch: float) -> str:
    return f"ch{int(ch):03d}" if ch == int(ch) else f"ch{ch:06.1f}"


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _get_connectors():
    from ..adapters.comix import ComixConnector
    return [ComixConnector()]


# ── Commands ─────────────────────────────────────────────────────


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
    from ..adapters.cli_hook import RichHook
    from .engine import Engine

    engine, config, paths = Engine.from_config()
    paths.output.mkdir(exist_ok=True)
    hook = RichHook()

    from ..adapters.local_source import LocalSource
    source = LocalSource(path) if path.is_dir() else _SingleFileSource(path)
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
    asyncio.run(_translate(input, source_lang, target_lang, from_ch, to_ch, force))


@app.callback()
def main(ctx: typer.Context):
    """Manga/manhwa translation pipeline."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_interactive())


# ── Interactive TUI ──────────────────────────────────────────────


async def _interactive():
    from ..adapters.tui import TUI, load_projects
    from ..config import load_config as _load_cfg

    _, paths = _load_cfg()
    paths.ensure()

    tui = TUI(log_file=paths.cache / "last_run.log")
    projects, chapters_map = await load_projects()
    tui.set_projects(projects, chapters_map)
    tui.start()

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
            await _run_pipeline(tui, input_str, result.force, result.from_ch, result.to_ch, paths)
            await tui.wait_for_key()
            projects, chapters_map = await load_projects()
            tui.set_projects(projects, chapters_map)
    finally:
        tui.stop()


# ── Pipeline execution (shared by translate cmd + interactive) ───


async def _run_pipeline(hook, input_str, force, from_ch, to_ch, paths):
    """Run translation pipeline via AppService."""
    from .app.service import AppService
    from .app.workflows.project import ResumePolicy

    try:
        hook.log("[dim]Loading models…[/]")
        service = await AppService.create(paths.root)
        paths = service.paths

        download_q = None
        download_count = 0

        if _is_url(input_str):
            project_id, chapters, download_q, download_count, _ = await _resolve_url(
                input_str, service.store, "", "", from_ch, to_ch, paths, hook)
        else:
            project_id, chapters = await _resolve_path(
                Path(input_str), service.store, "", "", from_ch, to_ch, hook)

        total_chapters = len(chapters) + download_count
        if not total_chapters:
            hook.log("[red]No chapters to translate.[/]")
            await service.close()
            return

        project = await service.get_project(project_id)
        name = project["title"] if project else "unknown"
        lang = f"{project['source_lang']}→{project['target_lang']}" if project else ""
        provider = f"{service.config.translation.provider}/{service.config.translation.model}"

        # Update TUI header if it has those attrs
        if hasattr(hook, '_project_name'):
            hook._project_name = name
            hook._project_lang = lang
            hook._project_provider = provider
            hook._project_total = total_chapters

        hook.log(
            f"[bold]{service.config.translation.provider}[/] / {service.config.translation.model}  "
            f"[cyan]{name}[/] {total_chapters} chapters "
            f"{lang}" + (" [yellow]force[/]" if force else "")
        )

        project_out = paths.output / name.lower().replace(" ", "-")

        def _on_chapter(ch, pages):
            _save_pages(pages, project_out / _ch_label(ch))

        result = await service.translate_project(
            project_id=project_id,
            chapters=chapters or None,
            on_chapter=_on_chapter,
            policy=ResumePolicy(force=force),
            chapter_stream=download_q,
            total_hint=total_chapters,
            hook=hook,
        )

        hook.log(
            f"\n[bold green]✓ Done[/] — {result['done']} done, "
            f"{result['skipped']} skipped, {result['failed']} failed"
        )
        await service.close()
    except Exception as e:
        hook.log(f"[bold red]error:[/] {e}")


async def _translate(input_str, source_lang, target_lang, from_ch, to_ch, force):
    from ..adapters.cli_hook import RichHook
    from ..config import load_config as _load_cfg

    _, paths = _load_cfg()
    paths.ensure()

    hook = RichHook(log_file=paths.cache / "last_run.log")
    hook.start()
    try:
        await _run_pipeline(hook, input_str, force, from_ch, to_ch, paths)
    finally:
        hook.stop()


# ── Input resolution ─────────────────────────────────────────────


async def _resolve_url(url, store, source_lang, target_lang, from_ch, to_ch, paths, hook):
    from ..adapters.local_source import LocalSource
    from ..downloader import download_images

    connector = next((c for c in _get_connectors() if c.accepts(url)), None)
    if not connector:
        raise RuntimeError(f"No connector for: {url}")
    if not connector.is_authenticated():
        raise RuntimeError(f"✗ {connector.site_name} — not ready. Run: typoon auth {connector.site_name}")

    hook.log(f"[dim]Discovering from {connector.site_name}…[/]")
    info = await connector.discover(url)
    hook.log(f"[cyan]{info.suggested_title}[/] ({info.suggested_lang}) — {len(info.chapters)} chapters")

    project = await store.get_project_by_url(url)
    if project:
        project_id = project["id"]
    else:
        sl = source_lang or info.suggested_lang
        tl = target_lang or "vi"
        project_id = await store.add_project(title=info.suggested_title, source_lang=sl, target_lang=tl, source_url=url)
        hook.log(f"Created project [cyan]{info.suggested_title}[/] ({sl} → {tl})")

    selected = [ch for ch in info.chapters
                if (from_ch == 0 or ch.number >= from_ch) and (to_ch == 0 or ch.number <= to_ch)]

    series_dir = paths.projects / info.suggested_title.lower().replace(" ", "-")
    chapters: list[tuple[float, object]] = []
    to_download: list = []
    headers = connector.http_headers()

    for ch in selected:
        ch_dir = series_dir / _ch_label(ch.number)
        if ch_dir.exists() and any(ch_dir.iterdir()):
            chapters.append((ch.number, LocalSource(ch_dir)))
            await store.add_chapter(project_id, ch.number, local_path=str(ch_dir))
        else:
            to_download.append(ch)

    download_q = None
    download_fail_count = [0]
    if to_download:
        download_q = asyncio.Queue()
        asyncio.create_task(_download_worker(
            to_download, connector, series_dir, headers, store, project_id,
            download_q, hook, download_fail_count))

    return project_id, chapters, download_q, len(to_download), download_fail_count


async def _download_worker(chapters, connector, series_dir, headers, store, project_id, queue, hook, fail_count):
    from ..adapters.local_source import LocalSource
    from ..downloader import download_images

    for ch in chapters:
        try:
            ch_dir = series_dir / _ch_label(ch.number)
            hook.log(f"[dim]↓ downloading ch{ch.number}…[/]")
            urls = await connector.get_page_urls(ch)
            await download_images(urls, ch_dir, headers)
            await store.add_chapter(project_id, ch.number, local_path=str(ch_dir))
            await queue.put((ch.number, LocalSource(ch_dir)))
        except Exception as e:
            hook.log(f"[red]✗[/] download ch{ch.number}: {e}")
            fail_count[0] += 1
    await queue.put(None)


async def _resolve_path(path, store, source_lang, target_lang, from_ch, to_ch, hook):
    from ..adapters.local_source import LocalSource

    if _has_chapter_subdirs(path):
        project_name = path.name
        chapter_dirs = _discover_chapters(path)
        chapters = [
            (_parse_chapter_num(d.name), LocalSource(d))
            for d in chapter_dirs
            if (from_ch == 0 or _parse_chapter_num(d.name) >= from_ch)
            and (to_ch == 0 or _parse_chapter_num(d.name) <= to_ch)
        ]
    elif _has_images(path):
        project_name = path.parent.name
        chapters = [(_parse_chapter_num(path.name), LocalSource(path))]
    else:
        raise RuntimeError("No images or chapter subfolders found")

    project = await store.get_project_by_title(project_name)
    if project:
        project_id = project["id"]
    else:
        sl = source_lang or "ko"
        tl = target_lang or "vi"
        project_id = await store.add_project(title=project_name, source_lang=sl, target_lang=tl)
        hook.log(f"Created project [cyan]{project_name}[/] ({sl} → {tl})")

    for idx, source in chapters:
        local_path = str(source._path) if hasattr(source, '_path') else None
        await store.add_chapter(project_id, idx, local_path=local_path)

    return project_id, chapters


# ── Helpers ──────────────────────────────────────────────────────


def _save_pages(pages, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = [p for p in pages if p.rendered is not None]
    if not rendered:
        return 0
    for page in rendered:
        out = out_dir / f"p{page.index:03d}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(page.rendered, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        from PIL import Image
        pil_pages = [Image.fromarray(p.rendered) for p in rendered]
        pdf_path = out_dir / "chapter.pdf"
        pil_pages[0].save(str(pdf_path), "PDF", save_all=True,
                          append_images=pil_pages[1:], resolution=150)
    except ImportError:
        pass
    return len(rendered)


class _SingleFileSource:
    def __init__(self, path: Path):
        self._path = path
    async def fetch(self):
        pass
    def page_count(self) -> int:
        return 1
    def load_page(self, index: int):
        bgr = cv2.imread(str(self._path))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
