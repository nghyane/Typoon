"""Project management — CRUD and task enqueueing only.

No pipeline execution. No stage logic. Workers consume tasks from DB.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from typoon.domain.project import DiscoveredChapter, SourceInfo
from typoon.paths import ChapterPaths, Paths, ProjectPaths, slugify
from typoon.runs.events import (
    ChapterDownloaded, ChapterFailed, ChapterSkipped, Hook,
)
from typoon.sources.constants import IMAGE_EXTS
from typoon.storage.sqlite import SqliteStore


@dataclass(frozen=True)
class ChapterStatus:
    chapter_id:   int
    idx:          float
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


class Projects:
    def __init__(self, db: SqliteStore, paths: Paths) -> None:
        self._db    = db
        self._paths = paths

    @classmethod
    async def open(cls) -> "Projects":
        from typoon.config import load_config
        _config, paths = load_config()
        paths.ensure()
        return cls(await SqliteStore.open(paths.db), paths)

    async def close(self) -> None:
        await self._db.close()

    # ── Lookup ────────────────────────────────────────────────────────

    async def get_by_slug(self, slug: str) -> dict | None:
        for proj in await self._db.list_projects():
            if proj["slug"] == slug:
                return proj
        return None

    async def require(self, slug: str) -> dict:
        proj = await self.get_by_slug(slug)
        if proj is None:
            raise ValueError(f"Project '{slug}' not found. Run 'typoon status'.")
        return proj

    # ── Discover ──────────────────────────────────────────────────────

    async def discover(self, url: str) -> SourceInfo:
        return await _connector(url).discover(url)

    # ── Pull from remote URL ──────────────────────────────────────────

    async def pull_new(
        self,
        info: SourceInfo,
        url: str,
        selected: list[DiscoveredChapter],
        target_lang: str,
        hook: Hook,
    ) -> str:
        slug       = slugify(info.suggested_title, url)
        project_id = await self._db.get_or_create_project(
            slug=slug, title=info.suggested_title,
            source_lang=info.suggested_lang, target_lang=target_lang,
            source_url=url,
        )
        proj = await self._db.get_project(project_id)
        ProjectPaths(self._paths.projects, slug).ensure()
        await self._download_and_enqueue(proj, selected, _connector(url), hook)
        return slug

    async def pull_more(
        self,
        slug: str,
        url: str,
        selected: list[DiscoveredChapter],
        hook: Hook,
    ) -> None:
        proj = await self.require(slug)
        await self._download_and_enqueue(proj, selected, _connector(url), hook)

    # ── Import local folder ───────────────────────────────────────────

    async def import_new(
        self,
        folder: Path,
        title: str,
        source_lang: str,
        target_lang: str,
        hook: Hook,
    ) -> str:
        slug       = slugify(title)
        project_id = await self._db.get_or_create_project(
            slug=slug, title=title,
            source_lang=source_lang, target_lang=target_lang,
        )
        proj = await self._db.get_project(project_id)
        ProjectPaths(self._paths.projects, slug).ensure()
        await self._import_and_enqueue(proj, folder, hook)
        return slug

    async def import_more(self, slug: str, folder: Path, hook: Hook) -> None:
        proj = await self.require(slug)
        await self._import_and_enqueue(proj, folder, hook)

    # ── Enqueue translation ───────────────────────────────────────────

    async def enqueue_translate(
        self,
        slug: str,
        indices: list[float] | None = None,
        redo: str | None = None,
    ) -> None:
        proj     = await self.require(slug)
        chapters = await self._db.get_all_chapters(proj["id"])

        if indices is not None:
            idx_set  = set(indices)
            chapters = [c for c in chapters if c["idx"] in idx_set]

        for ch in chapters:
            if redo:
                await self._db.delete_tasks_from(ch["id"], redo)
            first_stage = redo or _next_stage(
                ProjectPaths(self._paths.projects, proj["slug"]).chapter(ch["id"])
            )
            if first_stage:
                await self._db.enqueue(ch["id"], first_stage)

    # ── Status ────────────────────────────────────────────────────────

    async def get_status(self, slug: str | None = None) -> list[ProjectStatus]:
        projects = await self._db.list_projects()
        if slug:
            projects = [p for p in projects if p["slug"] == slug]

        result = []
        for proj in projects:
            proj_paths = ProjectPaths(self._paths.projects, proj["slug"])
            chapters   = []
            for ch in await self._db.get_all_chapters(proj["id"]):
                cp           = proj_paths.chapter(ch["id"])
                render_count = len(list(cp.render.iterdir())) if cp.is_rendered else 0
                chapters.append(ChapterStatus(
                    chapter_id=ch["id"],
                    idx=ch["idx"],
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

    # ── Internal ──────────────────────────────────────────────────────

    async def _download_and_enqueue(
        self,
        proj: dict,
        selected: list[DiscoveredChapter],
        connector,
        hook: Hook,
    ) -> None:
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])

        for ch in selected:
            chapter_id = await self._db.get_or_create_chapter(
                proj["id"], ch.number, source_url=ch.best_variant.url
            )
            cp = proj_paths.chapter(chapter_id)
            cp.ensure()

            if cp.is_prepared:
                hook.on(ChapterSkipped(chapter_id=chapter_id, reason="images_exist"))
            else:
                try:
                    from typoon.downloader import download_images
                    page_urls = await connector.get_page_urls(ch)
                    await download_images(page_urls, cp.pages)
                    hook.on(ChapterDownloaded(chapter_id=chapter_id, page_count=len(page_urls)))
                except Exception as e:
                    hook.on(ChapterFailed(chapter_id=chapter_id, stage="download", error=e))
                    continue

            await self._db.enqueue(chapter_id, "scan")

    async def _import_and_enqueue(self, proj: dict, folder: Path, hook: Hook) -> None:
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])

        for i, src_dir in enumerate(_chapter_dirs(folder), start=1):
            ch_num     = _parse_idx(src_dir.name) or float(i)
            chapter_id = await self._db.get_or_create_chapter(proj["id"], ch_num)
            cp         = proj_paths.chapter(chapter_id)
            cp.ensure()

            if cp.is_prepared:
                hook.on(ChapterSkipped(chapter_id=chapter_id, reason="images_exist"))
            else:
                _copy_images(src_dir, cp.pages)
                hook.on(ChapterDownloaded(
                    chapter_id=chapter_id,
                    page_count=len(list(cp.pages.iterdir())),
                ))

            await self._db.enqueue(chapter_id, "scan")


# ── Helpers ───────────────────────────────────────────────────────────


def _connector(url: str):
    from typoon.sources.connectors import get_connectors
    c = next((c for c in get_connectors() if c.accepts(url)), None)
    if c is None:
        raise ValueError(f"No connector for URL: {url}")
    return c


def _next_stage(cp: ChapterPaths) -> str | None:
    if not cp.is_prepared: return "prepare"
    if not cp.is_scanned:  return "scan"
    return "translate"


def _chapter_dirs(folder: Path) -> list[Path]:
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
    for i, f in enumerate(sorted(
        f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )):
        shutil.copy2(f, dest / f"{i:04d}{f.suffix.lower()}")
