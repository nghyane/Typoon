"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio
from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException

from typoon.adapters.artifact_store import ArtifactStore, LocalArtifactStore
from typoon.adapters.event_bus import EventBus, is_postgres, make_event_bus
from typoon.api.auth import verify_jwt
from typoon.config import AuthConfig, Config, load_config
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


def get_auth_cfg() -> AuthConfig:
    cfg, _ = _config_and_paths()
    return cfg.auth


# ── Auth dependency ──────────────────────────────────────────────────


async def require_user(
    authorization: str | None = Header(None),
    db:   Store      = Depends(get_store),
    cfg:  AuthConfig = Depends(get_auth_cfg),
) -> dict:
    """All authenticated routes depend on this. 401 if missing/invalid."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization[7:].strip()
    try:
        user_id = verify_jwt(token, cfg=cfg)
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e

    user = await db.get_user(user_id)
    if user is None:
        raise HTTPException(401, "User not found")
    return user


async def require_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("tier") != "admin":
        raise HTTPException(403, "Admin only")
    return user
