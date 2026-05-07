"""Discord OAuth + JWT issuance.

Single endpoint: POST /api/auth/discord/exchange { code, redirect_uri }
Both web standalone and Discord Activity use it identically.

Web flow:
  1. /login page in the SPA builds the Discord authorize URL with
     redirect_uri = <web>/auth/callback and a CSRF state stored in
     sessionStorage.
  2. Discord redirects back to /auth/callback?code=...&state=...
  3. SPA verifies state, POSTs the code here, gets JWT.

Discord Activity flow:
  1. SDK.commands.authorize() returns code.
  2. Activity POSTs the code here with redirect_uri = the value the SDK
     used internally (also exposed by the SDK).

The engine never serves HTML or sets cookies for OAuth — it's a pure
JSON exchanger. State management lives in the SPA, where the storage
boundary is per-tab anyway.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.auth import exchange_code, fetch_user, is_guild_member, issue_jwt
from typoon.api.deps import get_auth_cfg, get_store, require_user
from typoon.config import AuthConfig
from typoon.storage import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


class ExchangeBody(BaseModel):
    code:         str  # OAuth authorization code from Discord
    redirect_uri: str  # must match the redirect_uri sent to Discord at authorize


@router.get("/config")
async def auth_config(cfg: AuthConfig = Depends(get_auth_cfg)):
    """Public auth config. The SPA builds the Discord authorize URL itself,
    so it needs the client_id + the configured guild gating hint. Secrets
    (client_secret, jwt_secret) are not exposed."""
    if not cfg.discord_client_id:
        raise HTTPException(503, "Discord OAuth not configured")
    return {
        "discord_client_id": cfg.discord_client_id,
        "guild_gated":       bool(cfg.discord_guild_id),
    }


@router.post("/discord/exchange")
async def discord_exchange(
    body: ExchangeBody,
    cfg:  AuthConfig = Depends(get_auth_cfg),
    db:   Store      = Depends(get_store),
):
    """Code → JWT. Used by both web standalone (/auth/callback page) and
    Discord Activity (after sdk.commands.authorize)."""
    token = await _exchange_and_issue(
        code=body.code, redirect_uri=body.redirect_uri, cfg=cfg, db=db,
    )
    return {"token": token}


@router.get("/me")
async def me(user: dict = Depends(require_user)):
    """Returns the current authenticated user."""
    return _user_out(user)


@router.post("/logout", status_code=204)
async def logout():
    """JWT is stateless; the client just discards the token. This endpoint
    exists so the UI can call a logical 'logout' route. If we add a token
    revocation list later, this is where it goes."""
    return None


# ── Internals ────────────────────────────────────────────────────────


async def _exchange_and_issue(
    *, code: str, redirect_uri: str, cfg: AuthConfig, db: Store,
) -> str:
    if not cfg.discord_client_id or not cfg.discord_client_secret:
        raise HTTPException(503, "Discord OAuth not configured")

    try:
        access_token = await exchange_code(code, redirect_uri=redirect_uri, cfg=cfg)
    except Exception as e:
        raise HTTPException(400, f"OAuth exchange failed: {e}") from e

    discord_user = await fetch_user(access_token)

    # Gate by guild membership. Empty guild_id → gating disabled.
    if cfg.discord_guild_id:
        in_guild = await is_guild_member(access_token, cfg.discord_guild_id)
        if not in_guild:
            raise HTTPException(
                403,
                "Bạn cần tham gia Discord guild của Typoon để truy cập.",
            )

    promote = (
        cfg.bootstrap_discord_id
        and discord_user.id == cfg.bootstrap_discord_id
    )
    user = await db.upsert_user_from_identity(
        provider="discord",
        external_id=discord_user.id,
        display_name=discord_user.display_name,
        avatar_url=discord_user.avatar_url,
        email=discord_user.email,
        metadata={
            "username":    discord_user.username,
            "global_name": discord_user.global_name,
            "verified":    discord_user.verified,
        },
        promote_admin=bool(promote),
    )
    return issue_jwt(user["id"], cfg=cfg)


def _user_out(row: dict) -> dict:
    return {
        "id":           row["id"],
        "display_name": row["display_name"],
        "avatar_url":   row.get("avatar_url"),
        "email":        row.get("email"),
        "tier":         row["tier"],
        "created_at":   row.get("created_at"),
        "last_login_at": row.get("last_login_at"),
    }
