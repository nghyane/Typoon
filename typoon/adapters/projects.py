"""Project management — CRUD and task enqueueing only.

No pipeline execution. No stage logic. Workers consume tasks from DB.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from typoon.adapters.artifact_store import ArtifactStore, LocalArtifactStore
from typoon.domain.project import DiscoveredChapter, SourceInfo
from typoon.paths import Paths, ProjectPaths, slugify
from typoon.runs.events import (
    ChapterDownloaded, ChapterFailed, ChapterSkipped, Hook,
)
from typoon.sources.constants import IMAGE_EXTS
from typoon.sources.local import LocalSource
from typoon.stages.prepare_archive import prepare_chapter_to_archive
from typoon.storage import Store, SqliteStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChapterStatus:
    chapter_id:   int
    idx:          float
    state:        str    # done | running | error | pending | idle
    stage:        str    # scan | translate | render | ""
    render_count: int
    error:        str    # last error message if state == error


@dataclass(frozen=True)
class ProjectStatus:
    project_id:  int
    slug:        str
    title:       str
    source_lang: str
    target_lang: str
    chapters:    tuple[ChapterStatus, ...]


class Projects:
    def __init__(self, db: Store, paths: Paths, store: ArtifactStore) -> None:
        self._db    = db
        self._paths = paths
        self._store = store

    @classmethod
    async def open(cls) -> "Projects":
        from typoon.config import load_config
        _config, paths = load_config()
        paths.ensure()
        return cls(
            await SqliteStore.open(paths.db),
            paths,
            LocalArtifactStore(paths.artifacts),
        )

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
        connector  = _connector(url)
        slug       = slugify(info.suggested_title, url)
        project_id = await self._db.get_or_create_project(
            slug=slug, title=info.suggested_title,
            source_lang=connector.source_lang, target_lang=target_lang,
            source_url=url,
        )
        proj_paths = ProjectPaths(self._paths.projects, slug)
        proj_paths.ensure()
        await self._apply_metadata(project_id, proj_paths, connector, info)
        proj = await self._db.get_project(project_id)
        await self._download_and_enqueue(proj, selected, connector, hook)
        return slug

    async def pull_more(
        self,
        slug: str,
        url: str,
        info: SourceInfo,
        selected: list[DiscoveredChapter],
        hook: Hook,
    ) -> None:
        proj       = await self.require(slug)
        proj_paths = ProjectPaths(self._paths.projects, proj["slug"])
        connector  = _connector(url)
        await self._apply_metadata(proj["id"], proj_paths, connector, info)
        await self._download_and_enqueue(proj, selected, connector, hook)

    async def _apply_metadata(
        self,
        project_id: int,
        proj_paths: ProjectPaths,
        connector,
        info: SourceInfo,
    ) -> None:
        """Patch title/description and download cover if missing."""
        await self._db.update_project_metadata(
            project_id,
            title=info.suggested_title,
            description=info.description,
        )
        if info.cover_url and not proj_paths.cover.exists():
            await _download_cover(info.cover_url, proj_paths, connector)
            await self._db.update_project_metadata(
                project_id, cover_path=str(proj_paths.cover)
            )

    # ── Import local folder ───────────────────────────────────────────

    async def import_new(
        self,
        folder: Path,
        title: str,
        source_lang: str,
        target_lang: str,
        hook: Hook,
    ) -> int:
        slug       = slugify(title)
        project_id = await self._db.get_or_create_project(
            slug=slug, title=title,
            source_lang=source_lang, target_lang=target_lang,
        )
        proj = await self._db.get_project(project_id)
        ProjectPaths(self._paths.projects, slug).ensure()
        await self._import_and_enqueue(proj, folder, hook)
        return project_id

    async def import_more(self, slug: str, folder: Path, hook: Hook) -> None:
        proj = await self.require(slug)
        await self._import_and_enqueue(proj, folder, hook)

    # ── Redo — full reset + re-enqueue ───────────────────────────────

    async def redo(
        self,
        slug: str,
        indices: list[float] | None = None,
    ) -> int:
        """Reset chapters' derived state and enqueue scan. Returns count.

        Prepared archive is left intact: scan re-derives geometry/masks from
        prepared pixels. delete_chapter_data also flips the `rendered` flag
        back to false so the UI stops showing the old render until the new
        one completes.
        """
        proj     = await self.require(slug)
        chapters = await self._db.get_all_chapters(proj["id"])

        if indices is not None:
            idx_set  = set(indices)
            chapters = [c for c in chapters if c["idx"] in idx_set]

        for ch in chapters:
            await self._db.delete_chapter_data(ch["id"])
            await self._db.enqueue(ch["id"], "scan")

        return len(chapters)

    # ── Status ────────────────────────────────────────────────────────

    async def get_status(self, slug: str | None = None) -> list[ProjectStatus]:
        projects = await self._db.list_projects()
        if slug:
            projects = [p for p in projects if p["slug"] == slug]

        result = []
        for proj in projects:
            chapters = []
            for ch in await self._db.get_chapters_with_status(proj["id"]):
                chapters.append(ChapterStatus(
                    chapter_id=ch["chapter_id"],
                    idx=ch["idx"],
                    state=ch["state"],
                    stage=ch.get("stage") or "",
                    render_count=int(ch.get("page_count") or 0) if ch["state"] == "done" else 0,
                    error=ch.get("error") or "",
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
        for ch in selected:
            chapter_id = await self._db.get_or_create_chapter(
                proj["id"], ch.number,
                source_url=ch.best_variant.url,
                title=ch.title,
            )
            existing = await self._db.get_chapter_render_state(chapter_id)
            if existing and existing["page_count"] > 0:
                hook.on(ChapterSkipped(
                    chapter_id=chapter_id, chapter_idx=ch.number,
                    project_id=proj["id"], reason="prepared_exists",
                ))
            else:
                try:
                    from typoon.downloader import download_images
                    page_urls = await connector.get_page_urls(ch)
                    with tempfile.TemporaryDirectory() as tmp:
                        await download_images(page_urls, Path(tmp))
                        _key, n = await prepare_chapter_to_archive(
                            LocalSource(Path(tmp)),
                            project_id=proj["id"], chapter_id=chapter_id,
                            store=self._store,
                        )
                    await self._db.set_prepared_done(chapter_id, n)
                    hook.on(ChapterDownloaded(
                        chapter_id=chapter_id, chapter_idx=ch.number,
                        project_id=proj["id"], page_count=n,
                    ))
                except Exception as e:
                    hook.on(ChapterFailed(
                        chapter_id=chapter_id, chapter_idx=ch.number,
                        project_id=proj["id"], stage="download", error=e,
                    ))
                    continue

            await self._db.enqueue(chapter_id, "scan")

    async def _import_and_enqueue(self, proj: dict, folder: Path, hook: Hook) -> None:
        for i, src_dir in enumerate(_chapter_dirs(folder), start=1):
            ch_num     = _parse_idx(src_dir.name) or float(i)
            chapter_id = await self._db.get_or_create_chapter(proj["id"], ch_num)

            existing = await self._db.get_chapter_render_state(chapter_id)
            if existing and existing["page_count"] > 0:
                hook.on(ChapterSkipped(
                    chapter_id=chapter_id, chapter_idx=ch_num,
                    project_id=proj["id"], reason="prepared_exists",
                ))
            else:
                try:
                    _key, n = await prepare_chapter_to_archive(
                        LocalSource(src_dir),
                        project_id=proj["id"], chapter_id=chapter_id,
                        store=self._store,
                    )
                    await self._db.set_prepared_done(chapter_id, n)
                    hook.on(ChapterDownloaded(
                        chapter_id=chapter_id, chapter_idx=ch_num,
                        project_id=proj["id"], page_count=n,
                    ))
                except Exception as e:
                    hook.on(ChapterFailed(
                        chapter_id=chapter_id, chapter_idx=ch_num,
                        project_id=proj["id"], stage="prepare", error=e,
                    ))
                    continue

            await self._db.enqueue(chapter_id, "scan")


# ── Helpers ───────────────────────────────────────────────────────────


def _connector(url: str):
    from typoon.sources.connectors import get_connectors
    c = next((c for c in get_connectors() if c.accepts(url)), None)
    if c is None:
        raise ValueError(f"No connector for URL: {url}")
    return c


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


async def _download_cover(
    cover_url: str,
    proj_paths: ProjectPaths,
    connector,
) -> None:
    """Download cover to projects/<slug>/cover.jpg. Logs and skips on failure."""
    import io
    import httpx
    from PIL import Image

    headers = connector.http_headers() if hasattr(connector, "http_headers") else {}
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30) as client:
            resp = await client.get(cover_url)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except (httpx.HTTPError, OSError) as e:
        logger.warning("cover download failed (%s): %s", cover_url, e)
        return
    proj_paths.ensure()
    img.save(proj_paths.cover, format="JPEG", quality=88)
