"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from typoon.api.events import EventBus, LocalEventBus, PostgresEventBus
from typoon.config import load_config
from typoon.paths import Paths
from typoon.storage import SqliteStore, Store

_store: Store | None = None
_bus:   EventBus | None = None


@lru_cache(maxsize=1)
def get_config_and_paths():
    return load_config()


def _is_postgres(url: str) -> bool:
    return url.startswith(("postgresql://", "postgres://"))


async def _create_store(database_url: str, db_path: Path) -> Store:
    if _is_postgres(database_url):
        raise NotImplementedError("PostgresStore — Phase 2")
    return await SqliteStore.open(db_path)


def _create_bus(database_url: str, store: Store) -> EventBus:
    if _is_postgres(database_url):
        return PostgresEventBus(database_url)
    return LocalEventBus(store)


async def get_store() -> Store:
    global _store
    if _store is None:
        config, paths = get_config_and_paths()
        _store = await _create_store(config.database_url, paths.db)
    return _store


async def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        config, _ = get_config_and_paths()
        store     = await get_store()
        _bus      = _create_bus(config.database_url, store)
    return _bus


def get_paths() -> Paths:
    _, paths = get_config_and_paths()
    return paths
