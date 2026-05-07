"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from typoon.adapters.artifact_store import ArtifactStore, LocalArtifactStore
from typoon.adapters.event_bus import EventBus, is_postgres, make_event_bus
from typoon.config import Config, load_config
from typoon.paths import Paths
from typoon.storage import Store

_store: Store | None = None
_bus:   EventBus | None = None
_artifact_store: ArtifactStore | None = None
_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def _config_and_paths() -> tuple[Config, Paths]:
    return load_config()


async def get_store() -> Store:
    global _store
    if _store is None:
        async with _lock:
            if _store is None:
                from typoon.storage import SqliteStore
                config, paths = _config_and_paths()
                if is_postgres(config.database_url):
                    raise NotImplementedError("PostgresStore — Phase 2")
                _store = await SqliteStore.open(paths.db)
    return _store


async def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        async with _lock:
            if _bus is None:
                config, _ = _config_and_paths()
                _bus = make_event_bus(config, await get_store())
    return _bus


def get_paths() -> Paths:
    _, paths = _config_and_paths()
    return paths


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _, paths = _config_and_paths()
        _artifact_store = LocalArtifactStore(paths.artifacts)
    return _artifact_store
