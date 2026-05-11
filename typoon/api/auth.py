"""Authentication — Discord OAuth + JWT bearer tokens.

Single auth model that works for:

  - Web (standalone): user redirects through Discord OAuth, browser stores
    JWT in localStorage, sends `Authorization: Bearer <jwt>` on each
    request.
  - Discord Activity: SDK gives an OAuth `code`, web exchanges it for the
    same JWT via the same /api/auth/discord/exchange endpoint.
  - Discord bot / browser extension (later): same Bearer JWT, can be
    issued via a CLI/admin route once we add per-user API tokens.

Why JWT instead of cookie session: Discord Activities run in
{client_id}.discordsays.com sandbox, cookies require Partitioned +
SameSite=None plumbing. Bearer header is identical for web/DA/bot/ext;
no SameSite gymnastics.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

import httpx
import jwt

from typoon.config import AuthConfig

logger = logging.getLogger(__name__)

DISCORD_API   = "https://discord.com/api"
JWT_ALGORITHM = "HS256"


# ── DTOs ──────────────────────────────────────────────────────────────


@dataclass
class DiscordUser:
    id:           str           # snowflake (string in Discord JSON)
    username:     str
    global_name:  str | None
    avatar:       str | None    # hash; build URL with cdn.discordapp.com
    email:        str | None
    verified:     bool

    @property
    def display_name(self) -> str:
        return self.global_name or self.username

    @property
    def avatar_url(self) -> str | None:
        if not self.avatar:
            return None
        return f"https://cdn.discordapp.com/avatars/{self.id}/{self.avatar}.png"


# ── JWT ──────────────────────────────────────────────────────────────


def issue_jwt(user_id: int, *, cfg: AuthConfig, role_ids: list[str] | None = None) -> str:
    now = int(time.time())
    payload = {
        # PyJWT requires `sub` to be a string per RFC 7519. We store the
        # numeric user_id as its string repr; verify_jwt parses back to int.
        "sub":   str(user_id),
        "iat":   now,
        "exp":   now + cfg.session_days * 86400,
        "jti":   secrets.token_urlsafe(8),
        # Discord role IDs the user held at OAuth time. Snapshotted in
        # the token — role changes need a re-login. Trade-off: zero
        # server-side cache, no bot listener.
        "roles": role_ids or [],
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str, *, cfg: AuthConfig) -> tuple[int, list[str]]:
    """Returns (user_id, role_ids). Raises jwt.InvalidTokenError on failure."""
    payload = jwt.decode(token, cfg.jwt_secret, algorithms=[JWT_ALGORITHM])
    return int(payload["sub"]), list(payload.get("roles") or [])


# ── Discord OAuth ─────────────────────────────────────────────────────


async def exchange_code(code: str, *, redirect_uri: str, cfg: AuthConfig) -> str:
    """OAuth code → access_token. The redirect_uri must match what the
    SPA used at /oauth2/authorize — Discord rejects mismatches with 400.
    Raises RuntimeError on failure.
    """
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{DISCORD_API}/oauth2/token",
            data={
                "client_id":     cfg.discord_client_id,
                "client_secret": cfg.discord_client_secret,
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if r.status_code != 200:
            logger.warning("Discord token exchange failed: %s %s", r.status_code, r.text[:200])
            raise RuntimeError(f"Discord OAuth failed ({r.status_code})")
        return r.json()["access_token"]


async def fetch_user(access_token: str) -> DiscordUser:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{DISCORD_API}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        r.raise_for_status()
        d = r.json()
    return DiscordUser(
        id=d["id"],
        username=d["username"],
        global_name=d.get("global_name"),
        avatar=d.get("avatar"),
        email=d.get("email"),
        verified=bool(d.get("verified", False)),
    )


async def fetch_user_guilds(access_token: str) -> list[dict]:
    """Return the user's guild list from /users/@me/guilds.

    Each entry: { id, name, icon, owner, permissions, ... }. Empty list
    on failure (logged). Caller decides what to do with stale data.
    """
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{DISCORD_API}/users/@me/guilds",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if r.status_code != 200:
            logger.warning("/users/@me/guilds: %s %s", r.status_code, r.text[:200])
            return []
        return r.json()


async def fetch_guild_member(access_token: str, guild_id: str) -> dict | None:
    """Return /users/@me/guilds/{guild_id}/member.

    Requires OAuth scope `guilds.members.read`. Payload includes
    `roles: [snowflake, ...]` — role IDs the user holds in that guild.
    Returns None on 404 (not a member) or any non-200.
    """
    if not guild_id:
        return None
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{DISCORD_API}/users/@me/guilds/{guild_id}/member",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if r.status_code != 200:
            logger.info(
                "/users/@me/guilds/%s/member: %s", guild_id, r.status_code,
            )
            return None
        return r.json()


async def fetch_guild_widget(guild_id: str) -> dict | None:
    """Public guild metadata via the widget endpoint.

    Returns { id, name, instant_invite, members, ... } if the guild has
    "Enable Server Widget" turned on; None otherwise. No auth required.
    Used to surface a friendly error ('join <name>: <invite>') when a
    user fails the guild gate.

    Cached in-process for `_WIDGET_TTL_SECONDS` so every authenticated
    page load doesn't hit Discord. Widget data (name + invite link)
    changes rarely enough that 5 minutes of staleness is fine, and the
    cache hit takes /api/auth/me from ~140ms down to ~5ms — the spinner
    on app boot blocked on this round trip on every refresh.
    """
    if not guild_id:
        return None
    now = time.time()
    cached = _WIDGET_CACHE.get(guild_id)
    if cached is not None and now - cached[0] < _WIDGET_TTL_SECONDS:
        return cached[1]
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{DISCORD_API}/guilds/{guild_id}/widget.json")
        except httpx.HTTPError as e:
            logger.warning("widget fetch error: %s", e)
            return None
    if r.status_code != 200:
        logger.info("widget disabled or guild private (%s) for %s", r.status_code, guild_id)
        # Negative-cache so a misconfigured guild doesn't trigger a
        # Discord round trip on every signed-in request.
        _WIDGET_CACHE[guild_id] = (now, None)
        return None
    data = r.json()
    _WIDGET_CACHE[guild_id] = (now, data)
    return data


_WIDGET_TTL_SECONDS = 300.0
_WIDGET_CACHE: dict[str, tuple[float, dict | None]] = {}


# OAuth URL builder lives in the SPA now — web/src/lib/auth.ts — because
# the SPA owns the redirect_uri (web origin /auth/callback) and CSRF
# state lifecycle. This module is a pure code→JWT exchanger.
