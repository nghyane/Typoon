"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from typoon.api.events import EventBus
from typoon.config import Config, load_config
from typoon.paths import Paths
from typoon.storage import Store

_store: Store | None = None
_bus:   EventBus | None = None
_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def _config_and_paths() -> tuple[Config, Paths]:
    return load_config()


def _is_postgres(url: str) -> bool:
    return url.startswith(("postgresql://", "postgres://"))


async def get_store() -> Store:
    global _store
    if _store is None:
        async with _lock:
            if _store is None:
                from typoon.storage import SqliteStore
                config, paths = _config_and_paths()
                if _is_postgres(config.database_url):
                    raise NotImplementedError("PostgresStore — Phase 2")
                _store = await SqliteStore.open(paths.db)
    return _store


async def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        async with _lock:
            if _bus is None:
                from typoon.api.events import LocalEventBus, PostgresEventBus
                config, _ = _config_and_paths()
                url = config.database_url
                if _is_postgres(url):
                    _bus = PostgresEventBus(url)
                else:
                    _bus = LocalEventBus(await get_store())
    return _bus


def get_paths() -> Paths:
    _, paths = _config_and_paths()
    return paths
