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


async def fetch_guild_member_roles(access_token: str, guild_id: str) -> list[str]:
    """Discord role IDs the user holds in `guild_id`, via the OAuth
    user token (requires `guilds.members.read` scope at authorize).

    Returns `[]` when the user is not a member of the guild (404),
    when the bot/app has no member-read access (403), or when the
    guild_id is empty. Never raises for these expected paths — RBAC
    is opt-in on top of identity, not a precondition for login.

    Other HTTP errors raise so the caller can decide whether to fail
    the OAuth exchange.
    """
    if not guild_id:
        return []
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{DISCORD_API}/users/@me/guilds/{guild_id}/member",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if r.status_code in (403, 404):
        # 404: user is not in the configured guild.
        # 403: the OAuth scope `guilds.members.read` wasn't granted, or
        #      the guild has restricted member fetches. Either way,
        #      treat as "no roles" — user logs in without admin.
        return []
    if r.status_code != 200:
        logger.warning(
            "Discord guild member fetch failed: %s %s",
            r.status_code, r.text[:200],
        )
        r.raise_for_status()
    roles = r.json().get("roles") or []
    # Discord returns snowflakes as strings; keep them that way to match
    # AuthConfig.admin_role_id (also string) so equality comparison in
    # require_admin is type-safe without coercion.
    return [str(rid) for rid in roles]


# OAuth URL builder lives in the SPA now — web/src/features/auth/session.ts —
# because the SPA owns the redirect_uri (web origin /auth/callback) and CSRF
# state lifecycle. This module is a pure code→JWT exchanger.
