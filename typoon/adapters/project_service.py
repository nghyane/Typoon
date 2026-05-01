"""ProjectService — single entry point for all project operations.

CLI calls service with a Hook. Service emits events via hook.on().
CLI implements Hook to render events. No isinstance chains anywhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.session import Session, make_session
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.paths import ChapterPaths, Paths, ProjectPaths, slugify
from typoon.runs.events import (
    ChapterDone, ChapterDownloaded, ChapterFailed,
    ChapterSkipped, Hook, StageDone, StageStarted,
)
from typoon.storage.sqlite import SqliteStore


class ProjectService:
    """Orchestrates discovery, download, and translation pipeline."""

    def __init__(
        self,
        db: SqliteStore,
        paths: Paths,
        runtime: VisionRuntime,
        config=None,
    ) -> None:
        self._db = db
        self._paths = paths
        self._runtime = runtime
        self._config = config

    @classmethod
    async def open(cls) -> "ProjectService":
        from typoon.config import load_config
        paths = Paths()
        paths.ensure()
        config, _ = load_config()
        runtime, config, _ = VisionRuntime.from_config(config)
        db = await SqliteStore.open(paths.db)
        return cls(db, paths, runtime, config)

    async def close(self) -> None:
        await self._db.close()

    # ── discover ──────────────────────────────────────────────────────

    async def discover(self, url: str):
        """Fetch chapter list. Returns SourceInfo."""
        return await self._get_connector(url).discover(url)

    # ── pull ──────────────────────────────────────────────────────────

    async def pull(
        self,
        url: str,
        selected_chapters: list,
        target_lang: str,
        hook: Hook,
        redo_from: str | None = None,
    ) -> None:
        """Download and translate selected chapters. Events emitted via hook."""
        connector = self._get_connector(url)
        info = await connector.discover(url)
        slug = slugify(info.suggested_title, url)
        proj_paths = ProjectPaths(self._paths.projects, slug)
        proj_paths.ensure()

        project_id = await self._db.get_or_create_project(
            slug=slug,
            title=info.suggested_title,
            source_lang=info.suggested_lang,
            target_lang=target_lang,
            source_url=url,
        )
        proj = await self._db.get_project(project_id)

        for ch in selected_chapters:
            cp = proj_paths.chapter(ch.number)
            cp.ensure()

            if cp.pages.exists() and any(cp.pages.iterdir()):
                hook.on(ChapterSkipped(idx=ch.number, reason="images_exist"))
            else:
                page_urls = await connector.get_page_urls(ch)
                from typoon.downloader import download_images
                await download_images(page_urls, cp.pages)
                await self._db.add_chapter(project_id, ch.number, source_url=ch.best_variant.url)
                hook.on(ChapterDownloaded(idx=ch.number, page_count=len(page_urls)))

            await self._run_chapter(cp, proj, hook, redo_from)

    # ── add ───────────────────────────────────────────────────────────

    async def add(
        self,
        folder: Path,
        title: str,
        source_lang: str,
        target_lang: str,
        hook: Hook,
        redo_from: str | None = None,
    ) -> None:
        """Import local folder and run pipeline. Events via hook."""
        slug = slugify(title)
        proj_paths = ProjectPaths(self._paths.projects, slug)
        proj_paths.ensure()

        project_id = await self._db.get_or_create_project(
            slug=slug, title=title,
            source_lang=source_lang, target_lang=target_lang,
        )
        proj = await self._db.get_project(project_id)

        for i, src_dir in enumerate(_detect_chapters(folder), start=1):
            ch_num = _parse_ch_num(src_dir.name) or float(i)
            cp = proj_paths.chapter(ch_num)
            cp.ensure()

            if cp.pages.exists() and any(cp.pages.iterdir()):
                hook.on(ChapterSkipped(idx=ch_num, reason="images_exist"))
            else:
                _copy_images(src_dir, cp.pages)
                await self._db.add_chapter(project_id, ch_num)
                hook.on(ChapterDownloaded(idx=ch_num, page_count=len(list(cp.pages.iterdir()))))

            await self._run_chapter(cp, proj, hook, redo_from)

    # ── translate ─────────────────────────────────────────────────────

    async def translate(
        self,
        project_id: int,
        chapter_indices: list[float],
        hook: Hook,
        redo_from: str | None = None,
    ) -> None:
        """Run pipeline for specific chapters. Events via hook."""
        proj = await self._db.get_project(project_id)
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])
        for idx in chapter_indices:
            cp = proj_paths.chapter(idx)
            cp.ensure()
            await self._run_chapter(cp, proj, hook, redo_from)

    # ── status ────────────────────────────────────────────────────────

    async def get_status(self) -> list[dict]:
        projects = await self._db.list_projects()
        result = []
        for proj in projects:
            proj_paths = ProjectPaths(self._paths.projects, proj["slug"])
            chapters = await self._db.get_all_chapters(proj["id"])
            ch_status = []
            for ch in chapters:
                cp = proj_paths.chapter(ch["idx"])
                render_count = len(list(cp.render.iterdir())) if cp.is_rendered else 0
                ch_status.append({**ch, "render_count": render_count, "cp": cp})
            result.append({**proj, "chapters": ch_status})
        return result

    # ── internal ──────────────────────────────────────────────────────

    async def _run_chapter(
        self,
        cp: ChapterPaths,
        proj: dict,
        hook: Hook,
        redo_from: str | None,
    ) -> None:
        from typoon.stages.pipeline import (
            _redo_from, _stage_prepare, _stage_scan,
            _stage_translate, _stage_render, _already_done,
        )

        _redo_from(cp, redo_from)
        idx = _idx_from_cp(cp)
        await self._db.set_chapter_status(proj["id"], idx, "translating")

        try:
            session = make_session(
                project_id=proj["id"],
                chapter=idx,
                source_lang=proj["source_lang"],
                target_lang=proj["target_lang"],
                store=self._db,
                config=self._config,
            )

            for stage_name, stage_fn in [
                ("prepare",   lambda: _stage_prepare(cp, None)),
                ("scan",      lambda: _stage_scan(cp, self._runtime, None)),
                ("translate", lambda: _stage_translate(cp, session, None)),
                ("render",    lambda: _stage_render(cp, self._runtime, None)),
            ]:
                if _already_done(cp, stage_name):
                    continue
                hook.on(StageStarted(idx=idx, stage=stage_name))
                result = stage_fn()
                if hasattr(result, "__await__"):
                    await result
                hook.on(StageDone(idx=idx, stage=stage_name))

            await self._db.set_chapter_status(proj["id"], idx, "done")

            bubble_count = 0
            if cp.is_translated:
                from typoon.domain.translate import Chapter as TC
                bubble_count = len(TC.load(cp).all_bubbles)

            hook.on(ChapterDone(idx=idx, bubble_count=bubble_count, render_dir=str(cp.render)))

        except Exception as e:
            await self._db.set_chapter_status(proj["id"], idx, "error")
            hook.on(ChapterFailed(idx=idx, stage="pipeline", error=e))

    def _get_connector(self, url: str):
        from typoon.adapters.connectors import get_connectors
        connector = next((c for c in get_connectors() if c.accepts(url)), None)
        if connector is None:
            raise ValueError(f"No connector for URL: {url}")
        return connector


# ── Helpers ───────────────────────────────────────────────────────────


def _detect_chapters(folder: Path) -> list[Path]:
    _IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if any(f.suffix.lower() in _IMG for f in folder.iterdir() if f.is_file()):
        return [folder]
    return sorted(
        d for d in folder.iterdir()
        if d.is_dir() and any(f.suffix.lower() in _IMG for f in d.iterdir() if f.is_file())
    )


def _parse_ch_num(name: str) -> float | None:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def _copy_images(src: Path, dest: Path) -> None:
    _IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in _IMG)
    for i, f in enumerate(files):
        shutil.copy2(f, dest / f"{i + 1:03d}{f.suffix.lower()}")


def _idx_from_cp(cp: ChapterPaths) -> float:
    import re
    m = re.match(r"ch(\d+(?:\.\d+)?)", cp.root.name)
    return float(m.group(1)) if m else 0.0
