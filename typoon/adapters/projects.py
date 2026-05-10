"""Project management — CRUD, redo, status, and local ingestion.

This adapter no longer scrapes remote sites. Chapters arrive as local
files (folder, zip/cbz, PDF, individual images) — see `sources.upload`
for the extraction pipeline. Workers consume scan/translate/render
tasks from the DB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from typoon.adapters.chapter_archive import masks_key
from typoon.adapters.storage_registry import StorageRegistry
from typoon.paths import Paths, ProjectPaths, slugify
from typoon.runs.events import (
    ChapterDownloaded, ChapterFailed, ChapterSkipped, Hook,
)
from typoon.sources.constants import IMAGE_EXTS
from typoon.sources.local import LocalSource
from typoon.stages.prepare_archive import prepare_chapter_to_archive
from typoon.storage import PostgresStore, Store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChapterStatus:
    chapter_id:   int
    number:       str
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
    def __init__(self, db: Store, paths: Paths, stores: StorageRegistry) -> None:
        self._db       = db
        self._paths    = paths
        # Pipeline holds prepared.bnl + masks.npz; readers map dispatches
        # public render archive deletes through whichever backend wrote them.
        self._stores   = stores
        self._pipeline = stores.pipeline

    @classmethod
    async def open(cls) -> "Projects":
        from typoon.adapters.storage_registry import build_storage
        from typoon.config import load_config
        config, paths = load_config()
        paths.ensure()
        stores = build_storage(config, paths)
        return cls(
            await PostgresStore.open(config.database_url),
            paths,
            stores,
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

    # ── Local folder import (CLI: `typoon add`) ───────────────────────

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

    # ── Single chapter ingest (API: upload endpoint) ──────────────────

    async def ingest_chapter(
        self,
        project_id: int,
        number: str,
        source_dir: Path,
        *,
        title: str | None = None,
        hook: Hook | None = None,
        strategy: str = "auto",
        start: bool = False,
    ) -> int:
        """Pack pages from `source_dir` as a chapter and (optionally)
        enqueue the scan stage.

        `source_dir` must contain a flat list of image files in reading
        order — callers (HTTP upload, CLI add) are responsible for
        unpacking archives or rendering PDFs into that shape first.

        `number` is the display string ("4", "4.5", "Extra"). The server
        derives the internal `position` from it. Returns the chapter_id.
        `strategy` is forwarded to `prepare_chapter_to_archive`
        (auto / one_to_one / stitch).

        `start=False` (default) leaves the chapter in `idle` so the
        caller can review or trigger later via `start_chapters`. Set
        `start=True` for callers that already committed to running the
        pipeline (CLI `add`, extension import).
        """
        chapter_id = await self._db.create_chapter(
            project_id, number, title=title,
        )
        try:
            _key, n = await prepare_chapter_to_archive(
                LocalSource(source_dir),
                project_id=project_id, chapter_id=chapter_id,
                store=self._pipeline,
                strategy=strategy,
            )
        except Exception as e:
            if hook is not None:
                hook.on(ChapterFailed(
                    chapter_id=chapter_id, chapter_number=number,
                    project_id=project_id, stage="prepare", error=e,
                ))
            raise

        await self._db.set_prepared_done(chapter_id, n)
        if hook is not None:
            hook.on(ChapterDownloaded(
                chapter_id=chapter_id, chapter_number=number,
                project_id=project_id, page_count=n,
            ))
        if start:
            await self._db.enqueue(chapter_id, "scan")
        return chapter_id

    # ── Manual trigger ────────────────────────────────────────────────

    async def start_chapters(
        self, project_id: int, chapter_ids: list[int],
    ) -> int:
        """Enqueue `scan` for idle chapters. Returns count actually
        started.

        Skips:
          - chapters already in flight (pending/running) — already on
            their way, re-enqueue is a no-op.
          - done/error chapters — caller must use `redo()` to reset
            derived data first.
          - unknown ids / ids from another project.
        """
        started = 0
        for cid in chapter_ids:
            ch = await self._db.get_chapter_with_status(cid, project_id)
            if ch is None or ch["state"] != "idle":
                continue
            await self._db.enqueue(cid, "scan")
            started += 1
        return started

    # ── Redo — full reset + re-enqueue ───────────────────────────────

    async def redo(
        self,
        slug: str,
        chapter_ids: list[int] | None = None,
    ) -> int:
        """Reset chapters' derived state and enqueue scan. Returns count.

        Prepared archive is left intact: scan re-derives geometry/masks from
        prepared pixels. delete_chapter_data also flips the `rendered` flag
        back to false so the UI stops showing the old render until the new
        one completes.
        """
        proj     = await self.require(slug)
        chapters = await self._db.get_all_chapters(proj["id"])

        if chapter_ids is not None:
            wanted = set(chapter_ids)
            chapters = [c for c in chapters if c["id"] in wanted]

        for ch in chapters:
            # Drop derived blobs BEFORE the DB row is reset — once
            # `delete_chapter_data` nulls archive_backend/locator we
            # lose the dispatch info for the old render archive.
            await self._purge_derived_blobs(
                proj["id"], ch["id"],
                ch.get("archive_backend"), ch.get("archive_locator"),
            )
            await self._db.delete_chapter_data(ch["id"])
            await self._db.enqueue(ch["id"], "scan")

        return len(chapters)

    async def _purge_derived_blobs(
        self,
        project_id: int,
        chapter_id: int,
        archive_backend: str | None,
        archive_locator: str | None,
    ) -> None:
        """Delete masks.npz + the public render archive.

        prepared.bnl is intentionally kept: scan re-derives geometry/masks
        from the same prepared pixels, so dropping it would force a
        re-upload for no gain. masks.npz must go because scan rewrites it
        from scratch and a stale copy would render against the old
        geometry if scan failed mid-stage. The public render archive
        also must go: redo invalidates it and leaving the blob behind
        creates an orphan on the public store.
        """
        await self._pipeline.delete(masks_key(project_id, chapter_id))
        if archive_backend and archive_locator:
            try:
                await self._stores.reader(archive_backend).delete(archive_locator)
            except RuntimeError:
                # Backend no longer configured — orphan, nothing to do.
                pass

    # ── Status (CLI) ──────────────────────────────────────────────────

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
                    number=ch["number"],
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

    async def _import_and_enqueue(self, proj: dict, folder: Path, hook: Hook) -> None:
        for i, src_dir in enumerate(_chapter_dirs(folder), start=1):
            number     = _parse_number(src_dir.name) or str(i)
            chapter_id = await self._db.create_chapter(proj["id"], number)

            existing = await self._db.get_chapter_render_state(chapter_id)
            if existing and existing["page_count"] > 0:
                hook.on(ChapterSkipped(
                    chapter_id=chapter_id, chapter_number=number,
                    project_id=proj["id"], reason="prepared_exists",
                ))
            else:
                try:
                    _key, n = await prepare_chapter_to_archive(
                        LocalSource(src_dir),
                        project_id=proj["id"], chapter_id=chapter_id,
                        store=self._pipeline,
                    )
                    await self._db.set_prepared_done(chapter_id, n)
                    hook.on(ChapterDownloaded(
                        chapter_id=chapter_id, chapter_number=number,
                        project_id=proj["id"], page_count=n,
                    ))
                except Exception as e:
                    hook.on(ChapterFailed(
                        chapter_id=chapter_id, chapter_number=number,
                        project_id=proj["id"], stage="prepare", error=e,
                    ))
                    continue

            await self._db.enqueue(chapter_id, "scan")


# ── Helpers ───────────────────────────────────────────────────────────


def _chapter_dirs(folder: Path) -> list[Path]:
    if any(f.suffix.lower() in IMAGE_EXTS for f in folder.iterdir() if f.is_file()):
        return [folder]
    return sorted(
        d for d in folder.iterdir()
        if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
    )


def _parse_number(name: str) -> str | None:
    """Extract the leading numeric token from a folder name.

    Returns the matched string ("12", "4.5") or None if no digit was
    found. Callers fall back to a positional counter when None.
    """
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return m.group(1) if m else None
