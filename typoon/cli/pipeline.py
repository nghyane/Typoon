"""CLI pipeline execution — shared by translate command and interactive TUI."""

from __future__ import annotations

from pathlib import Path

from ..app.service import AppService
from ..app.workflows.project import ResumePolicy
from .output import save_pages
from .resolve import resolve_path, resolve_url
from .utils import ch_label, is_url


async def run_pipeline(hook, input_str, force, from_ch, to_ch, paths):
    """Run translation pipeline via AppService."""
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
            return

        project = await service.get_project(project_id)
        name = project["title"] if project else ppaths.slug if ppaths else "unknown"
        lang = f"{project['source_lang']}→{project['target_lang']}" if project else ""
        provider = f"{service.config.translation.provider}/{service.config.translation.model}"

        if hasattr(hook, '_project_name'):
            hook._project_name = name
            hook._project_lang = lang
            hook._project_provider = provider
            hook._project_total = total_chapters

        hook.log(
            f"[bold]{service.config.translation.provider}[/] / {service.config.translation.model}  "
            f"[cyan]{name}[/] {total_chapters} chapters {lang}"
            + (" [yellow]force[/]" if force else "")
        )

        out_root = ppaths.output_dir if ppaths else paths.output

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

        hook.log(
            f"\n[bold green]✓ Done[/] — {result['done']} done, "
            f"{result['skipped']} skipped, {result['failed']} failed"
        )
        await service.close()
    except Exception as e:
        hook.log(f"[bold red]error:[/] {e}")


async def translate_cli(input_str, source_lang, target_lang, from_ch, to_ch, force):
    """CLI translate command wrapper."""
    from .hook import RichHook
    from ..config import load_config as _load_cfg

    _, paths = _load_cfg()
    paths.ensure()

    hook = RichHook(log_file=paths.cache / "last_run.log")
    hook.start()
    try:
        await run_pipeline(hook, input_str, force, from_ch, to_ch, paths)
    finally:
        hook.stop()
