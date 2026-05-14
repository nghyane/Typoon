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

Discord is the identity provider AND the authorization source. Login
itself has no guild gate — anyone who can OAuth gets a session. RBAC
is layered on top: at exchange time we read the user's role IDs in
`AuthConfig.discord_guild_id` (scope `guilds.members.read`) and embed
them in the JWT. `require_admin` then checks for
`AuthConfig.admin_role_id` membership. Users outside the guild log in
with `roles=[]` and are treated as ordinary members.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.auth import exchange_code, fetch_guild_member_roles, fetch_user, issue_jwt
from typoon.api.deps import get_auth_cfg, get_store, require_user
from typoon.api.models import SessionUser
from typoon.config import AuthConfig
from typoon.storage import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


class ExchangeBody(BaseModel):
    code:         str  # OAuth authorization code from Discord
    redirect_uri: str  # must match the redirect_uri sent to Discord at authorize


@router.get("/config")
async def auth_config(cfg: AuthConfig = Depends(get_auth_cfg)):
    """Public auth config the SPA needs at /login.

    Refuses to start a login flow when Discord OAuth isn't configured;
    we no longer gate on guild membership so there's nothing else to
    check up front.
    """
    if not cfg.discord_client_id:
        raise HTTPException(
            503,
            "Discord OAuth chưa được cấu hình. "
            "Set DISCORD_CLIENT_ID + DISCORD_CLIENT_SECRET.",
        )
    return {
        "discord_client_id": cfg.discord_client_id,
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


@router.get("/me", response_model=SessionUser)
async def me(
    user: dict       = Depends(require_user),
    cfg:  AuthConfig = Depends(get_auth_cfg),
) -> SessionUser:
    """Canonical session payload. The SPA caches this under the
    React Query key `['session']`; every consumer (header avatar,
    settings page, reading-language picker) reads from that one
    cache instead of refetching."""
    return _session_user(user, cfg=cfg)


@router.post("/logout", status_code=204)
async def logout():
    """JWT is stateless; the client just discards the token. This endpoint
    exists so the UI can call a logical 'logout' route. If we add a token
    revocation list later, this is where it goes."""
    return None


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

    # Snapshot the user's guild role IDs into the JWT so `require_admin`
    # can authorise without a per-request Discord round trip. Role
    # changes therefore need a re-login — acceptable for an ops surface
    # where elevation is a deliberate, infrequent event.
    role_ids = await fetch_guild_member_roles(access_token, cfg.discord_guild_id)

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
    )

    return issue_jwt(user_id=user["id"], role_ids=role_ids, cfg=cfg)


def _session_user(user: dict, *, cfg: AuthConfig) -> SessionUser:
    """Project a Store user row into the wire-shape `SessionUser`,
    deriving `is_admin` from the JWT-snapshotted Discord roles.
    Shared by `GET /api/auth/me` and `PATCH /api/me/preferences`
    so both endpoints emit byte-identical payloads — the SPA can
    overwrite the session cache with either response without
    schema-shape branching."""
    is_admin = (
        bool(cfg.admin_role_id)
        and cfg.admin_role_id in user.get("roles", [])
    )
    return SessionUser(
        id                    = user["id"],
        display_name          = user["display_name"],
        avatar_url            = user.get("avatar_url"),
        is_admin              = is_admin,
        preferred_target_lang = user.get("preferred_target_lang"),
    )
