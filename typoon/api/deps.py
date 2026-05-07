"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio
from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException

from typoon.adapters.artifact_store import ArtifactStore, LocalArtifactStore
from typoon.adapters.event_bus import EventBus
from typoon.api.auth import verify_jwt
from typoon.config import AuthConfig, Config, load_config
from typoon.paths import Paths
from typoon.storage import PostgresStore, Store

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
                config, _ = _config_and_paths()
                _store = await PostgresStore.open(config.database_url)
    return _store


async def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        async with _lock:
            if _bus is None:
                config, _ = _config_and_paths()
                _bus = EventBus(config.database_url)
    return _bus


def get_paths() -> Paths:
    _, paths = _config_and_paths()
    return paths


def get_config() -> Config:
    cfg, _ = _config_and_paths()
    return cfg


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
    """All authenticated routes depend on this. 401 if missing/invalid.

    Attaches `roles` (list of Discord role IDs from the JWT) onto the
    user dict so downstream `require_admin`/route checks can read them
    without re-decoding the token.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization[7:].strip()
    try:
        user_id, role_ids = verify_jwt(token, cfg=cfg)
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e

    user = await db.get_user(user_id)
    if user is None:
        raise HTTPException(401, "User not found")
    user["roles"] = role_ids
    return user


async def require_admin(
    user: dict = Depends(require_user),
    cfg:  AuthConfig = Depends(get_auth_cfg),
) -> dict:
    """Admin = user holds the configured Discord role ID."""
    if not cfg.admin_role_id:
        raise HTTPException(503, "Admin role not configured (DISCORD_ADMIN_ROLE_ID)")
    if cfg.admin_role_id not in user.get("roles", []):
        raise HTTPException(403, "Admin only")
    return user
