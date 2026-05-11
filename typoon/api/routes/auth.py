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

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.auth import (
    exchange_code, fetch_guild_member, fetch_guild_widget,
    fetch_user, fetch_user_guilds, issue_jwt,
)
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
    """Public auth config the SPA needs at /login.

    Refuses to start a login flow when the deployment is misconfigured.
    Better the operator sees a 503 with fix instructions than the SPA
    silently rendering with a fake brand name.
    """
    if not cfg.discord_client_id:
        raise HTTPException(503, "Discord OAuth chưa được cấu hình.")

    # Open mode (no gating). Brand is None — the SPA renders a generic,
    # unbranded shell.
    if not cfg.discord_guild_id:
        return {
            "discord_client_id":  cfg.discord_client_id,
            "guild_gated":        False,
            "guild_name":         None,
            "guild_icon_url":     None,
            "discord_invite_url": None,
        }

    # Gated: widget must expose at least the guild name. Without it the
    # SPA has nothing to brand the login page with, and gating-failure
    # users see no name in the 403 message either.
    name, invite = await _resolve_guild_invite(cfg.discord_guild_id)
    if not name:
        raise HTTPException(
            503,
            "Discord Server Widget chưa được bật. "
            "Discord Server Settings → Widget → Enable Server Widget.",
        )
    return {
        "discord_client_id":  cfg.discord_client_id,
        "guild_gated":        True,
        "guild_name":         name,
        "guild_icon_url":     _read_guild_icon(cfg.discord_guild_id),
        "discord_invite_url": invite,
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
async def me(
    user: dict = Depends(require_user),
    cfg:  AuthConfig = Depends(get_auth_cfg),
):
    name, _ = await _resolve_guild_invite(cfg.discord_guild_id) if cfg.discord_guild_id else (None, None)
    is_admin = bool(cfg.admin_role_id) and cfg.admin_role_id in user.get("roles", [])
    return {
        **_user_out(user, is_admin=is_admin),
        "guild_name":     name,
        "guild_icon_url": _read_guild_icon(cfg.discord_guild_id),
    }


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

    role_ids: list[str] = []

    # Gate by guild membership. Empty guild_id → gating disabled.
    if cfg.discord_guild_id:
        guilds = await fetch_user_guilds(access_token)
        match  = next((g for g in guilds if g.get("id") == cfg.discord_guild_id), None)
        if match is None:
            raise HTTPException(403, await _gate_message(cfg.discord_guild_id))
        # Capture guild icon for branding (widget doesn't expose it).
        _persist_guild_meta(cfg.discord_guild_id, icon=match.get("icon"))

        # Pull role IDs the user holds in this guild. Requires the
        # `guilds.members.read` OAuth scope. When missing or 404, fall
        # back to empty roles — user is still a member, just no admin
        # privileges.
        member = await fetch_guild_member(access_token, cfg.discord_guild_id)
        if member:
            role_ids = [str(r) for r in member.get("roles", []) if r]

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
    return issue_jwt(user["id"], cfg=cfg, role_ids=role_ids)


def _user_out(row: dict, *, is_admin: bool = False) -> dict:
    return {
        "id":            row["id"],
        "display_name":  row["display_name"],
        "avatar_url":    row.get("avatar_url"),
        "email":         row.get("email"),
        "is_admin":      is_admin,
        "created_at":    row.get("created_at"),
        "last_login_at": row.get("last_login_at"),
    }


async def _resolve_guild_invite(guild_id: str) -> tuple[str | None, str | None]:
    """Read (name, invite) from the public widget endpoint.

    Returns (None, None) when gating is disabled OR the guild has not
    enabled 'Server Widget'. Cached for 5 minutes in `fetch_guild_widget`
    so /api/auth/me doesn't pay a Discord round-trip on every request.
    """
    if not guild_id:
        return None, None
    widget = await fetch_guild_widget(guild_id)
    if not widget:
        return None, None
    return widget.get("name"), widget.get("instant_invite")


# ── Guild icon caching ───────────────────────────────────────────────
#
# Discord's widget endpoint exposes the guild name + invite but not the
# icon hash. The hash *is* in /users/@me/guilds, which we already call
# during login. We cache it to disk on the first guild-gated login so
# /api/auth/config can return guild_icon_url without an authenticated
# Discord call (and so unauthenticated /login can show the icon).

def _meta_path() -> Path:
    from typoon.api.deps import _config_and_paths
    _, paths = _config_and_paths()
    return paths.root / ".guild_meta.json"


def _persist_guild_meta(guild_id: str, *, icon: str | None) -> None:
    if not icon:
        return
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"guild_id": guild_id, "icon": icon}
    path.write_text(json.dumps(data))


def _read_guild_icon(guild_id: str) -> str | None:
    """Build the CDN URL for the cached icon, or None if we don't have one
    yet (no admin has logged in since the engine started fresh)."""
    if not guild_id:
        return None
    path = _meta_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("guild_id") != guild_id or not data.get("icon"):
        return None
    return f"https://cdn.discordapp.com/icons/{guild_id}/{data['icon']}.png?size=128"


async def _gate_message(guild_id: str) -> str:
    """Build a user-friendly 403 message. Pulls guild name + invite from
    the public widget endpoint; the operator only needs to enable
    'Server Widget' once in Discord Server Settings."""
    name, invite = await _resolve_guild_invite(guild_id)
    if invite and name:
        return f"Bạn cần tham gia Discord {name}: {invite}"
    if invite:
        return f"Bạn cần tham gia Discord guild: {invite}"
    if name:
        return f"Bạn cần tham gia Discord {name} để truy cập."
    # Widget disabled or guild private — admin needs to enable widget.
    return (
        "Bạn cần tham gia Discord guild để truy cập. "
        "(Quản trị viên: bật Server Widget trong Discord để hiển thị invite.)"
    )
