"""User self-service endpoints — identity, API tokens, quota.

The /projects alias is gone — its replacement is /api/library and the
list of the user's own translations is implicit (each user's
translations table rows). Discord guild memberships are exposed here
so the SPA can resolve current activity context (scope_guild_id).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from typoon.api.auth_token import issue_api_token
from typoon.api.deps import get_auth_cfg, get_config, get_store, require_user
from typoon.api.models import GuildOut, MeOut
from typoon.api.quota import get_quota_snapshot
from typoon.config import AuthConfig, Config
from typoon.storage import Store


router = APIRouter(
    prefix="/api/me", tags=["me"],
    dependencies=[Depends(require_user)],
)


# ── Request bodies ────────────────────────────────────────────────────


class CreateTokenBody(BaseModel):
    name: str = Field(min_length=1, max_length=64)


# ── Response models ──────────────────────────────────────────────────


class TokenInfo(BaseModel):
    id:         int
    name:       str
    prefix:     str
    last_used:  str | None = None
    created_at: str | None = None


class TokenCreated(TokenInfo):
    """Includes plaintext exactly once. UI must display it then drop."""
    token: str


# ── /me ──────────────────────────────────────────────────────────────


@router.get("", response_model=MeOut)
async def me(
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Current user + cached guild memberships.

    Guild list drives the visibility scope picker in the spawn modal
    and the Hội Mê Truyện guild picker in the browse hub. Refreshed
    by the auth/exchange endpoint at login; this read is a simple
    cache lookup.
    """
    guilds = await db.get_user_guilds(user["id"])
    return MeOut(
        id=user["id"],
        display_name=user["display_name"],
        avatar_url=user.get("avatar_url"),
        guilds=[
            GuildOut(
                id=g["id"], name=g.get("name"), icon_url=g.get("icon_url"),
            )
            for g in guilds
        ],
    )


# ── API tokens (CLI / extension) ─────────────────────────────────────


@router.get("/tokens", response_model=list[TokenInfo])
async def list_tokens(
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    rows = await db.list_api_tokens(user["id"])
    return [TokenInfo(**r) for r in rows]


@router.post("/tokens", response_model=TokenCreated, status_code=201)
async def create_token(
    body: CreateTokenBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Create a new API token. The plaintext appears in the response
    once and is never retrievable again — caller must persist it on
    the spot.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    token_id, plaintext, _prefix = await issue_api_token(
        db, user_id=user["id"], name=name,
    )
    rows = await db.list_api_tokens(user["id"])
    row = next((r for r in rows if r["id"] == token_id), None)
    if row is None:
        raise HTTPException(500, "token created but lookup failed")
    return TokenCreated(token=plaintext, **row)


@router.delete("/tokens/{token_id}", status_code=204)
async def revoke_token(
    token_id: int,
    user:     dict  = Depends(require_user),
    db:       Store = Depends(get_store),
):
    if not await db.revoke_api_token(user["id"], token_id):
        raise HTTPException(404, "Token not found")


# ── Quota ────────────────────────────────────────────────────────────


@router.get("/quota")
async def get_quota(
    user: dict       = Depends(require_user),
    db:   Store      = Depends(get_store),
    cfg:  Config     = Depends(get_config),
    auth: AuthConfig = Depends(get_auth_cfg),
):
    """Per-user translation quota snapshot for the sidebar widget."""
    return await get_quota_snapshot(user, db, cfg.rate_limit, auth)
