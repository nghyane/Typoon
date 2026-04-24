"""CLI pipeline execution for the `translate` command."""

from __future__ import annotations

from pathlib import Path

from ..app.service import AppService
from ..app.workflows.project import ResumePolicy
from .output import save_pages
from .resolve import resolve_path, resolve_url
from .utils import ch_label, is_url


async def run_pipeline(hook, input_str, force, from_ch, to_ch, paths):
    """Run translation pipeline via AppService.

    Returns a summary dict {done, skipped, failed, out_root, name} or
    None if nothing was run. Caller prints the summary after tearing
    down any transient UI, so output location survives in scrollback.
    """
    try:
        hook.log("[dim]Loading models…[/]")
        service = await AppService.create(paths.root)
        paths = service.paths

        download_q = None
        download_count = 0
        ppaths = None

        if is_url(input_str):
            project_id, chapters, download_q, download_count, _, ppaths = await resolve_url(
                input_str, service.store, "", "", from_ch, to_ch, paths, hook)
        else:
            project_id, chapters, ppaths = await resolve_path(
                Path(input_str), service.store, "", "", from_ch, to_ch, paths, hook)

        total_chapters = len(chapters) + download_count
        if not total_chapters:
            hook.log("[red]No chapters to translate.[/]")
            await service.close()
            return None

        project = await service.get_project(project_id)
        name = project["title"] if project else ppaths.slug if ppaths else "unknown"
        lang = f"{project['source_lang']}→{project['target_lang']}" if project else ""

        hook.log(
            f"[bold]{service.config.translation.provider}[/] / {service.config.translation.model}  "
            f"[cyan]{name}[/] {total_chapters} chapters {lang}"
            + (" [yellow]force[/]" if force else "")
        )

        out_root = ppaths.output_dir if ppaths else paths.output
        hook.log(f"[dim]Output → {out_root}[/]")

        def _on_chapter(ch, pages):
            save_pages(pages, out_root / ch_label(ch))

        result = await service.translate_project(
            project_id=project_id,
            chapters=chapters or None,
            on_chapter=_on_chapter,
            policy=ResumePolicy(force=force),
            chapter_stream=download_q,
            total_hint=total_chapters,
            hook=hook,
        )

        await service.close()
        return {**result, "out_root": out_root, "name": name}
    except Exception as e:
        hook.log(f"[bold red]error:[/] {e}")
        return None


async def translate_cli(input_str, source_lang, target_lang, from_ch, to_ch, force):
    """CLI translate command wrapper."""
    from rich.console import Console

    from ..config import load_config as _load_cfg
    from .hook import CliHook

    _, paths = _load_cfg()
    paths.ensure()

    log_file = paths.cache / "last_run.log"
    hook = CliHook(log_file=log_file)
    hook.start()
    try:
        summary = await run_pipeline(hook, input_str, force, from_ch, to_ch, paths)
    finally:
        hook.stop()

    console = Console()
    if summary is None:
        console.print(f"[dim]No output. See log:[/] [cyan]{log_file}[/]")
        return
    console.print(
        f"\n[bold green]✓ Done[/] [cyan]{summary['name']}[/] — "
        f"{summary['done']} done, {summary['skipped']} skipped, "
        f"{summary['failed']} failed"
    )
    console.print(f"[bold]Output:[/] [cyan]{summary['out_root']}[/]")
    console.print(f"[dim]Log:[/]    [cyan]{log_file}[/]")
