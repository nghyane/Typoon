"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio
from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException

from typoon.adapters.channel_bus import ChannelBus
from typoon.adapters.inbox import ChapterInbox, build_inbox
from typoon.adapters.storage_registry import StorageRegistry, build_storage
from typoon.api.auth import verify_jwt
from typoon.api.auth_token import looks_like_api_token, verify_api_token
from typoon.config import AuthConfig, Config, load_config
from typoon.paths import Paths
from typoon.storage.postgres import PostgresStore
from typoon.storage.store    import Store

_store: Store | None = None
_bus:   ChannelBus | None = None
_storage: StorageRegistry | None = None
_inbox: ChapterInbox | None = None
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
                _store = await PostgresStore.open(
                    config.database_url,
                    pool_min_size=config.database.pool_min_size,
                    pool_max_size=config.database.pool_max_size,
                    statement_cache_size=config.database.statement_cache_size,
                )
    return _store


async def get_bus() -> ChannelBus:
    global _bus
    if _bus is None:
        async with _lock:
            if _bus is None:
                config, _ = _config_and_paths()
                _bus = ChannelBus(config.database_url)
    return _bus


def get_paths() -> Paths:
    _, paths = _config_and_paths()
    return paths


def get_config() -> Config:
    cfg, _ = _config_and_paths()
    return cfg


def get_storage() -> StorageRegistry:
    """Pipeline + public stores for this process.

    Workers use `storage.pipeline` to share prepared/masks across hosts.
    API URL build uses `storage.reader(row.archive_backend)` to
    dispatch chapter URLs through whichever public backend wrote them.

    Pre-warmed by the API lifespan hook so the first user-facing
    request never pays construction cost; a fallback double-check
    handles late binding from CLI / test contexts that skip the
    lifespan path. `build_storage` is sync and idempotent, so a
    rare double-init under raw concurrency leaks at most one extra
    registry instance (which holds no live file handles).
    """
    global _storage
    if _storage is None:
        cfg, paths = _config_and_paths()
        _storage = build_storage(cfg, paths)
    return _storage


def get_inbox() -> ChapterInbox:
    """Browser-facing chapter zip inbox (S3-compatible or local dev).

    Pre-warmed by the API lifespan hook (see `prewarm_singletons`),
    so concurrent first hits see a fully constructed object. Same
    idempotency caveat as `get_storage`.
    """
    global _inbox
    if _inbox is None:
        cfg, paths = _config_and_paths()
        _inbox = build_inbox(
            cfg.storage,
            paths_root=paths.artifacts,
            base_url=cfg.server.public_base_url,
        )
    return _inbox


async def prewarm_singletons() -> None:
    """Construct every lazily-initialised singleton once, from the
    API lifespan's single-threaded startup context. After this
    returns, subsequent `get_*` calls take the fast path with no
    chance of racing on construction."""
    await get_store()
    await get_bus()
    get_storage()
    get_inbox()


def get_auth_cfg() -> AuthConfig:
    cfg, _ = _config_and_paths()
    return cfg.auth


async def require_user(
    authorization: str | None = Header(None),
    db:   Store      = Depends(get_store),
    cfg:  AuthConfig = Depends(get_auth_cfg),
) -> dict:
    """All authenticated routes depend on this. 401 if missing/invalid.

    Accepts two credential shapes in the `Authorization: Bearer …`
    header:

    1. JWT (web SPA, Discord OAuth flow). Carries `roles` claim from
       Discord at login time → user["roles"] populated, used by
       `require_admin`.
    2. API token (`typ_…`). Issued via /api/me/tokens for
       extension/CLI/worker use. Token holders never get admin access:
       user["roles"] is set to []. Mutation routes (owner-only) still
       work via require_project_owner.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    raw = authorization[7:].strip()

    if looks_like_api_token(raw):
        user = await verify_api_token(db, raw)
        if user is None:
            raise HTTPException(401, "Invalid API token")
        user["roles"] = []
        return user

    try:
        user_id, role_ids = verify_jwt(raw, cfg=cfg)
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


async def require_worker(
    authorization: str | None = Header(None),
    db:   Store = Depends(get_store),
) -> dict:
    """Worker-only routes. Accepts API tokens whose scopes include
    'worker'. Distinct from `require_user` so JWT sessions and ordinary
    API tokens cannot reach pipeline blob endpoints.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    raw = authorization[7:].strip()
    if not looks_like_api_token(raw):
        raise HTTPException(401, "Worker scope requires API token")
    user = await verify_api_token(db, raw)
    if user is None:
        raise HTTPException(401, "Invalid API token")
    if "worker" not in (user.get("scopes") or []):
        raise HTTPException(403, "Worker scope required")
    return user
