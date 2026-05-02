"""Project orchestration — discovery, import, translation pipeline."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from typoon.adapters.ctx import make_ctx
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.project import DiscoveredChapter, SourceInfo
from typoon.paths import ChapterPaths, Paths, ProjectPaths, slugify
from typoon.runs.events import (
    ChapterDone, ChapterDownloaded, ChapterFailed,
    ChapterSkipped, Hook,
)
from typoon.sources.constants import IMAGE_EXTS
from typoon.storage.sqlite import SqliteStore


# ── Status dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class ChapterStatus:
    idx:          float
    status:       str
    render_count: int
    render_path:  str


@dataclass(frozen=True)
class ProjectStatus:
    project_id:  int
    slug:        str
    title:       str
    source_lang: str
    target_lang: str
    chapters:    tuple[ChapterStatus, ...]


class FatalError(Exception):
    """Infrastructure error that should stop the entire pipeline run."""


# ── Service ───────────────────────────────────────────────────────────


class ProjectService:
    """Orchestrates project creation, chapter import, and translation pipeline.

    Two modes for each import command:
      *_new  — create project from metadata, then import
      *_more — project already exists (identified by slug), just import

    Runtime lazy-loaded — status queries never load vision models.
    """

    def __init__(self, db: SqliteStore, paths: Paths, config=None) -> None:
        self._db = db
        self._paths = paths
        self._config = config
        self._runtime: VisionRuntime | None = None

    @property
    def runtime(self) -> VisionRuntime:
        if self._runtime is None:
            self._runtime, self._config, _ = VisionRuntime.from_config(self._config)
        return self._runtime

    @classmethod
    async def open(cls) -> "ProjectService":
        from typoon.config import load_config
        paths = Paths()
        paths.ensure()
        config, _ = load_config()
        db = await SqliteStore.open(paths.db)
        return cls(db, paths, config)

    async def close(self) -> None:
        await self._db.close()

    # ── Project lookup ────────────────────────────────────────────────

    async def get_project_by_slug(self, slug: str) -> dict | None:
        """Return project dict or None if not found."""
        for proj in await self._db.list_projects():
            if proj["slug"] == slug:
                return proj
        return None

    async def require_project(self, slug: str) -> dict:
        """Return project dict. Raises ValueError if not found."""
        proj = await self.get_project_by_slug(slug)
        if proj is None:
            raise ValueError(
                f"Project '{slug}' not found.\n"
                f"Run 'typoon status' to see existing projects."
            )
        return proj

    # ── discover ──────────────────────────────────────────────────────

    async def discover(self, url: str) -> SourceInfo:
        return await _connector(url).discover(url)

    # ── pull: create project from URL, then download ──────────────────

    async def pull_new(
        self,
        info: SourceInfo,
        url: str,
        selected: list[DiscoveredChapter],
        target_lang: str,
        hook: Hook,
        redo: str | None = None,
    ) -> str:
        """Create project from SourceInfo, download chapters. Returns slug."""
        slug = slugify(info.suggested_title, url)
        project_id = await self._db.get_or_create_project(
            slug=slug,
            title=info.suggested_title,
            source_lang=info.suggested_lang,
            target_lang=target_lang,
            source_url=url,
        )
        proj = await self._db.get_project(project_id)
        ProjectPaths(self._paths.projects, slug).ensure()
        await self._download_and_run(proj, selected, _connector(url), hook, redo)
        return slug

    # ── pull: add chapters to existing project ────────────────────────

    async def pull_more(
        self,
        slug: str,
        url: str,
        selected: list[DiscoveredChapter],
        hook: Hook,
        redo: str | None = None,
    ) -> None:
        """Download chapters into an existing project."""
        proj = await self.require_project(slug)
        await self._download_and_run(proj, selected, _connector(url), hook, redo)

    # ── add: create project from local folder ────────────────────────

    async def add_new(
        self,
        folder: Path,
        title: str,
        source_lang: str,
        target_lang: str,
        hook: Hook,
        redo: str | None = None,
    ) -> str:
        """Create project from local folder. Returns slug."""
        slug = slugify(title)
        project_id = await self._db.get_or_create_project(
            slug=slug, title=title,
            source_lang=source_lang, target_lang=target_lang,
        )
        proj = await self._db.get_project(project_id)
        ProjectPaths(self._paths.projects, slug).ensure()
        await self._import_folder(proj, folder, hook, redo)
        return slug

    # ── add: add chapters to existing project ────────────────────────

    async def add_more(
        self,
        slug: str,
        folder: Path,
        hook: Hook,
        redo: str | None = None,
    ) -> None:
        """Copy chapters from folder into an existing project."""
        proj = await self.require_project(slug)
        await self._import_folder(proj, folder, hook, redo)

    # ── translate ─────────────────────────────────────────────────────

    async def translate(
        self,
        slug: str,
        indices: list[float] | None,
        hook: Hook,
        redo: str | None = None,
    ) -> None:
        """Run pipeline for pending (or specified) chapters of a project."""
        proj = await self.require_project(slug)
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])

        if indices is None:
            chapters = await self._db.get_all_chapters(proj["id"])
            indices = [c["idx"] for c in chapters if c["status"] == "pending"]

        for idx in indices:
            cp = proj_paths.chapter(idx)
            cp.ensure()
            await self._run_chapter(cp, proj, hook, redo)

    # ── status ────────────────────────────────────────────────────────

    async def get_status(self, slug: str | None = None) -> list[ProjectStatus]:
        """Return status for all projects or a specific one. No model loading."""
        projects = await self._db.list_projects()
        if slug:
            projects = [p for p in projects if p["slug"] == slug]

        result = []
        for proj in projects:
            proj_paths = ProjectPaths(self._paths.projects, proj["slug"])
            chapters = []
            for ch in await self._db.get_all_chapters(proj["id"]):
                cp = proj_paths.chapter(ch["idx"])
                render_count = len(list(cp.render.iterdir())) if cp.is_rendered else 0
                chapters.append(ChapterStatus(
                    idx=ch["idx"],
                    status=ch["status"],
                    render_count=render_count,
                    render_path=str(cp.render),
                ))
            result.append(ProjectStatus(
                project_id=proj["id"],
                slug=proj["slug"],
                title=proj["title"],
                source_lang=proj["source_lang"],
                target_lang=proj["target_lang"],
                chapters=tuple(chapters),
            ))
        return result

    # ── internal ──────────────────────────────────────────────────────

    async def _download_and_run(
        self,
        proj: dict,
        selected: list[DiscoveredChapter],
        connector,
        hook: Hook,
        redo: str | None,
    ) -> None:
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])

        for ch in selected:
            cp = proj_paths.chapter(ch.number)
            cp.ensure()
            await self._db.add_chapter(proj["id"], ch.number, source_url=ch.best_variant.url)

            if _has_pages(cp):
                hook.on(ChapterSkipped(idx=ch.number, reason="images_exist"))
            else:
                await self._db.set_chapter_status(proj["id"], ch.number, "downloading")
                try:
                    from typoon.downloader import download_images
                    page_urls = await connector.get_page_urls(ch)
                    await download_images(page_urls, cp.pages)
                    hook.on(ChapterDownloaded(idx=ch.number, page_count=len(page_urls)))
                except Exception as e:
                    await self._db.set_chapter_status(proj["id"], ch.number, "error")
                    hook.on(ChapterFailed(idx=ch.number, stage="download", error=e))
                    continue

            await self._run_chapter(cp, proj, hook, redo)

    async def _import_folder(
        self,
        proj: dict,
        folder: Path,
        hook: Hook,
        redo: str | None,
    ) -> None:
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])

        for i, src_dir in enumerate(_chapter_dirs(folder), start=1):
            ch_num = _parse_idx(src_dir.name) or float(i)
            cp = proj_paths.chapter(ch_num)
            cp.ensure()
            await self._db.add_chapter(proj["id"], ch_num)

            if _has_pages(cp):
                hook.on(ChapterSkipped(idx=ch_num, reason="images_exist"))
            else:
                _copy_images(src_dir, cp.pages)
                hook.on(ChapterDownloaded(idx=ch_num, page_count=len(list(cp.pages.iterdir()))))

            await self._run_chapter(cp, proj, hook, redo)

    async def _run_chapter(
        self,
        cp: ChapterPaths,
        proj: dict,
        hook: Hook,
        redo: str | None,
    ) -> None:
        from typoon.stages import pipeline

        await self._db.set_chapter_status(proj["id"], cp.idx, "translating")
        ctx = make_ctx(
            project_id=proj["id"],
            chapter=cp.idx,
            source_lang=proj["source_lang"],
            target_lang=proj["target_lang"],
            store=self._db,
            config=self._config,
            hook=hook,
        )
        try:
            await pipeline.run(cp, ctx, self.runtime, hook=hook, redo=redo)
            await self._db.set_chapter_status(proj["id"], cp.idx, "done")
            hook.on(ChapterDone(idx=cp.idx, bubble_count=_count_bubbles(cp), render_dir=str(cp.render)))
        except Exception as e:
            await self._db.set_chapter_status(proj["id"], cp.idx, "error")
            hook.on(ChapterFailed(idx=cp.idx, stage="pipeline", error=e))
            if _is_fatal(e):
                raise FatalError(str(e)) from e


# ── Helpers ───────────────────────────────────────────────────────────


def _connector(url: str):
    from typoon.sources.connectors import get_connectors
    c = next((c for c in get_connectors() if c.accepts(url)), None)
    if c is None:
        raise ValueError(f"No connector for URL: {url}")
    return c


def _has_pages(cp: ChapterPaths) -> bool:
    return cp.pages.exists() and any(cp.pages.iterdir())


def _chapter_dirs(folder: Path) -> list[Path]:
    """Return list of image directories. Single chapter or multi-chapter."""
    if any(f.suffix.lower() in IMAGE_EXTS for f in folder.iterdir() if f.is_file()):
        return [folder]
    return sorted(
        d for d in folder.iterdir()
        if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
    )


def _parse_idx(name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def _copy_images(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    for i, f in enumerate(files):
        shutil.copy2(f, dest / f"{i + 1:03d}{f.suffix.lower()}")


def _count_bubbles(cp: ChapterPaths) -> int:
    if not cp.is_translated:
        return 0
    from typoon.domain.translate import Chapter as TC
    return len(TC.load(cp).all_bubbles)


def _is_fatal(exc: Exception) -> bool:
    """True for infrastructure errors that affect all chapters — stop immediately."""
    try:
        import openai
        if isinstance(exc, (
            openai.AuthenticationError,   # no API key / bad key
            openai.PermissionDeniedError, # no access to model
            openai.APIConnectionError,    # network unreachable
        )):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 403, 503):
            return True
    except ImportError:
        pass
    try:
        import anthropic
        if isinstance(exc, (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.APIConnectionError,
        )):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code in (401, 403, 503):
            return True
    except ImportError:
        pass
    return False
