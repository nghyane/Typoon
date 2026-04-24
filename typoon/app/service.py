"""AppService — single entry point for all operations.

No UI calls Engine/Store/Scanner directly. Everything goes through here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .events import CompositeSink, EventSink, Hook, HookAdapter
from .workflows.project import ResumePolicy, translate_project

if TYPE_CHECKING:
    from ..config import Config, Paths
    from ..engine import Engine
    from ..ports import ChapterSource, Store
    from ..domain.bubble import Page


class AppService:
    """Single entry point for all operations."""

    def __init__(
        self,
        config: Config,
        paths: Paths,
        store: Store,
        engine: Engine,
    ) -> None:
        self.config = config
        self.paths = paths
        self.store = store
        self.engine = engine
        self._sinks: list[EventSink] = []

    @classmethod
    async def create(cls, root: Path | None = None) -> AppService:
        """Bootstrap everything: config, DB, models."""
        from ..adapters.sqlite_store import SqliteStore
        from ..config import load_config
        from ..engine import Engine

        config, paths = load_config(root)
        paths.ensure()
        engine, config, paths = Engine.from_config(config=config, paths=paths)
        store = await SqliteStore.open(paths.db)
        return cls(config=config, paths=paths, store=store, engine=engine)

    def subscribe(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def _hook(self, override: Hook | None = None) -> Hook:
        """Return override hook, or build one from subscribed sinks."""
        if override is not None:
            return override
        if not self._sinks:
            return Hook()
        if len(self._sinks) == 1:
            return HookAdapter(self._sinks[0])
        return HookAdapter(CompositeSink(*self._sinks))

    # ── Project management ──

    async def get_project(self, project_id: int) -> dict | None:
        return await self.store.get_project(project_id)

    async def create_project(
        self, title: str, source_lang: str = "ko",
        target_lang: str = "vi", source_url: str | None = None,
    ) -> int:
        return await self.store.add_project(
            title=title, source_lang=source_lang,
            target_lang=target_lang, source_url=source_url)

    # ── Batch operations ──

    async def translate_project(
        self,
        project_id: int,
        chapters: list[tuple[float, ChapterSource]] | None = None,
        on_chapter: Callable[[float, list[Page]], None] | None = None,
        policy: ResumePolicy | None = None,
        chapter_stream: asyncio.Queue | None = None,
        total_hint: int = 0,
        hook: Hook | None = None,
    ) -> dict:
        return await translate_project(
            engine=self.engine, store=self.store, config=self.config,
            project_id=project_id, chapters=chapters,
            hook=self._hook(hook), on_chapter=on_chapter,
            policy=policy, chapter_stream=chapter_stream,
            total_hint=total_hint)

    # ── Review/edit ──

    async def update_bubble(
        self, project_id: int, chapter_idx: float,
        bubble_id: str, text: str,
    ) -> None:
        parts = bubble_id.replace("p", "").replace("b", " ").split("_")
        page, idx = int(parts[0]), int(parts[1])
        await self.store._db.execute(
            "UPDATE translations SET translated_text=? "
            "WHERE project_id=? AND chapter=? AND page=? AND idx=?",
            (text, project_id, chapter_idx, page, idx))
        await self.store._db.commit()

    # ── Cleanup ──

    async def close(self) -> None:
        await self.store.close()
