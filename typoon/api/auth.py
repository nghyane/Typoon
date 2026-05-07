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


def issue_jwt(user_id: int, *, cfg: AuthConfig) -> str:
    now = int(time.time())
    payload = {
        # PyJWT requires `sub` to be a string per RFC 7519. We store the
        # numeric user_id as its string repr; verify_jwt parses back to int.
        "sub": str(user_id),
        "iat": now,
        "exp": now + cfg.session_days * 86400,
        "jti": secrets.token_urlsafe(8),
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str, *, cfg: AuthConfig) -> int:
    """Returns user_id. Raises jwt.InvalidTokenError on failure."""
    payload = jwt.decode(token, cfg.jwt_secret, algorithms=[JWT_ALGORITHM])
    return int(payload["sub"])


# ── Discord OAuth ─────────────────────────────────────────────────────


async def exchange_code(code: str, *, cfg: AuthConfig) -> str:
    """OAuth code → access_token. Raises on failure."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{DISCORD_API}/oauth2/token",
            data={
                "client_id":     cfg.discord_client_id,
                "client_secret": cfg.discord_client_secret,
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  cfg.discord_redirect_uri,
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


async def is_guild_member(access_token: str, guild_id: str) -> bool:
    """Check whether the user is a member of the gating guild.

    Uses /users/@me/guilds (cheap, returns the user's guild list). Caller
    is responsible for caching the result; we don't want to hit Discord
    on every request.
    """
    if not guild_id:
        return True  # gating disabled
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{DISCORD_API}/users/@me/guilds",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if r.status_code != 200:
            logger.warning("guilds check failed: %s", r.status_code)
            return False
        guilds = r.json()
    return any(g.get("id") == guild_id for g in guilds)


# ── OAuth URL builder (for web standalone) ────────────────────────────


def build_authorize_url(
    *, cfg: AuthConfig, state: str, scope: str = "identify email guilds",
) -> str:
    """Standalone-web entry. Discord Activity uses SDK.authorize() instead."""
    from urllib.parse import urlencode
    params = {
        "client_id":     cfg.discord_client_id,
        "redirect_uri":  cfg.discord_redirect_uri,
        "response_type": "code",
        "scope":         scope,
        "state":         state,
        "prompt":        "consent",
    }
    return f"{DISCORD_API}/oauth2/authorize?{urlencode(params)}"
