"""CLI input resolution — URL → project/chapters or local path → project/chapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..adapters.local_source import LocalSource
from ..downloader import download_images
from .cli_utils import ch_label, is_url


def _get_connectors():
    from ..adapters.comix import ComixConnector
    return [ComixConnector()]


async def resolve_url(url, store, source_lang, target_lang, from_ch, to_ch, paths, hook):
    """URL → (project_id, chapters, download_q, download_count, fail_count)."""
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
        project_id = await store.add_project(
            title=info.suggested_title, source_lang=sl, target_lang=tl, source_url=url)
        hook.log(f"Created project [cyan]{info.suggested_title}[/] ({sl} → {tl})")

    selected = [
        ch for ch in info.chapters
        if (from_ch is None or from_ch == 0 or ch.number >= from_ch)
        and (to_ch is None or to_ch == 0 or ch.number <= to_ch)
    ]

    series_dir = paths.projects / info.suggested_title.lower().replace(" ", "-")
    chapters: list[tuple[float, object]] = []
    to_download: list = []
    headers = connector.http_headers()

    for ch in selected:
        ch_dir = series_dir / ch_label(ch.number)
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
    for ch in chapters:
        try:
            ch_dir = series_dir / ch_label(ch.number)
            hook.log(f"[dim]↓ downloading ch{ch.number}…[/]")
            urls = await connector.get_page_urls(ch)
            await download_images(urls, ch_dir, headers)
            await store.add_chapter(project_id, ch.number, local_path=str(ch_dir))
            await queue.put((ch.number, LocalSource(ch_dir)))
        except Exception as e:
            hook.log(f"[red]✗[/] download ch{ch.number}: {e}")
            fail_count[0] += 1
    await queue.put(None)


async def resolve_path(path, store, source_lang, target_lang, from_ch, to_ch, hook):
    """Local path → (project_id, chapters)."""
    from .cli_utils import discover_chapters, has_chapter_subdirs, has_images, parse_chapter_num

    if has_chapter_subdirs(path):
        project_name = path.name
        chapter_dirs = discover_chapters(path)
        chapters = [
            (parse_chapter_num(d.name), LocalSource(d))
            for d in chapter_dirs
            if (from_ch is None or from_ch == 0 or parse_chapter_num(d.name) >= from_ch)
            and (to_ch is None or to_ch == 0 or parse_chapter_num(d.name) <= to_ch)
        ]
    elif has_images(path):
        project_name = path.parent.name
        chapters = [(parse_chapter_num(path.name), LocalSource(path))]
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
