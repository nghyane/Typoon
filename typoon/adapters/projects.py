"""Project management — CRUD, redo, status, and local ingestion.

This adapter no longer scrapes remote sites. Chapters arrive as a flat
folder of images: the HTTP upload route unpacks the inbox zip first,
the CLI walks a local folder. Workers consume scan/translate/render
tasks from the DB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from typoon.adapters.chapter_archive import masks_key
from typoon.adapters.storage_registry import StorageRegistry
from typoon.paths import Paths
from typoon.runs.events import Hook
from typoon.storage import PostgresStore, Store

if TYPE_CHECKING:
    from typoon.adapters.inbox import InboxHandle

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

    # ── Single chapter queue (API: upload-finalize) ───────────────────

    async def queue_chapter(
        self,
        project_id: int,
        number: str,
        handle: "InboxHandle",
        *,
        title: str | None = None,
    ) -> int:
        """Create a chapter row + persist the inbox handle + enqueue
        prepare. Returns chapter_id.

        The HTTP upload-finalize route returns 202 the moment this
        function completes — it does NOT wait for prepare. The prepare
        worker reads back the inbox handle, downloads/unpacks the zip,
        runs `prepare_chapter_to_archive`, calls `set_prepared_done`,
        clears the inbox handle, and advances the task to `scan`.

        This is the only ingest path the engine accepts. The legacy
        CLI `typoon add` ran `prepare_chapter_to_archive` synchronously
        from a local folder; that command was removed when uploads
        moved to the multipart inbox flow.
        """
        chapter_id = await self._db.create_chapter(
            project_id, number, title=title,
        )
        # Bind the chapter id onto the handle so the worker can claim it.
        from dataclasses import replace
        bound = replace(handle, chapter_id=chapter_id, title=title)
        await self._db.set_inbox_handle(bound)
        await self._db.enqueue(chapter_id, "prepare")
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
    # (no internal helpers — chapter ingest goes through the public
    # `queue_chapter` API and the worker's prepare loop)
