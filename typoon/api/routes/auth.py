"""Discord OAuth + JWT issuance.

Two web entry shapes converge on /api/auth/discord/exchange:

  Web standalone:
    GET  /api/auth/discord/login         → 302 to Discord consent page
    GET  /api/auth/discord/callback      → exchanges code, returns HTML
                                            with embedded JWT (web reads it
                                            and stores in localStorage)
  Discord Activity:
    POST /api/auth/discord/exchange      → web posts SDK-issued code,
                                            same logic, JSON response

Both paths produce the same JWT and trigger the same gating + bootstrap
admin logic.
"""

from __future__ import annotations

import logging
import secrets
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from typoon.api.auth import (
    build_authorize_url, exchange_code, fetch_user, is_guild_member, issue_jwt,
)
from typoon.api.deps import get_auth_cfg, get_store, require_user
from typoon.config import AuthConfig
from typoon.storage import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

# State cookie used to defeat CSRF on the redirect-based flow. Signed by
# itsdangerous would be cleaner; for Phase 1 the value is short-lived
# enough that a 32-byte random nonce + httpOnly is sufficient.
_STATE_COOKIE = "typoon_oauth_state"
_STATE_MAX_AGE = 600  # 10 minutes


class ExchangeBody(BaseModel):
    code: str  # OAuth authorization code from Discord


@router.get("/discord/login")
async def discord_login(cfg: AuthConfig = Depends(get_auth_cfg)):
    """Standalone-web entry. Discord Activity uses POST /exchange directly."""
    if not cfg.discord_client_id:
        raise HTTPException(503, "Discord OAuth not configured")
    state = secrets.token_urlsafe(24)
    url = build_authorize_url(cfg=cfg, state=state)
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        _STATE_COOKIE, state,
        max_age=_STATE_MAX_AGE, httponly=True, samesite="lax", secure=False,
    )
    return response


@router.get("/discord/callback")
async def discord_callback(
    request: Request,
    code:  str | None = None,
    state: str | None = None,
    error: str | None = None,
    cfg:   AuthConfig = Depends(get_auth_cfg),
    db:    Store      = Depends(get_store),
):
    """Discord redirects here after consent. We exchange code for JWT and
    pass it to the web SPA via a tiny HTML bootstrap that stashes the
    token in localStorage and navigates to /."""
    if error:
        return _login_failed(f"Discord error: {error}", web_url=cfg.web_url)
    if not code or not state:
        return _login_failed("Missing code or state", web_url=cfg.web_url)

    expected_state = request.cookies.get(_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(state, expected_state):
        return _login_failed("State mismatch (possible CSRF)", web_url=cfg.web_url)

    try:
        token = await _exchange_and_issue(code=code, cfg=cfg, db=db)
    except HTTPException as e:
        return _login_failed(e.detail, web_url=cfg.web_url)
    except Exception as e:
        logger.exception("OAuth callback failed")
        return _login_failed(str(e), web_url=cfg.web_url)

    # Hand the JWT to the SPA via a one-shot bootstrap page. We avoid
    # putting it in the URL fragment because the web router would race
    # the storage write.
    html = _BOOTSTRAP_HTML.format(token=token, error="", web_url=cfg.web_url)
    response = HTMLResponse(html)
    response.delete_cookie(_STATE_COOKIE)
    return response


@router.post("/discord/exchange")
async def discord_exchange(
    body: ExchangeBody,
    cfg:  AuthConfig = Depends(get_auth_cfg),
    db:   Store      = Depends(get_store),
):
    """JSON exchange endpoint. Used by Discord Activity (SDK provides code)
    and by clients that prefer to handle the redirect themselves."""
    token = await _exchange_and_issue(code=body.code, cfg=cfg, db=db)
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


async def _exchange_and_issue(*, code: str, cfg: AuthConfig, db: Store) -> str:
    if not cfg.discord_client_id or not cfg.discord_client_secret:
        raise HTTPException(503, "Discord OAuth not configured")

    try:
        access_token = await exchange_code(code, cfg=cfg)
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


def _login_failed(reason: str, *, web_url: str) -> HTMLResponse:
    safe = reason.replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(
        _BOOTSTRAP_HTML.format(token="", error=quote(safe), web_url=web_url),
        status_code=200,
    )


# Tiny bootstrap page: stores the JWT (or error), then redirects to the
# web app. {web_url} comes from cfg.auth.web_url so the API and the SPA
# can run on different origins (Phase 1 dev: API :8000, web :5173).
_BOOTSTRAP_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Đang đăng nhập…</title>
<style>body{{margin:0;font:14px system-ui,sans-serif;color:#52525b;display:flex;align-items:center;justify-content:center;height:100vh;background:#fafafa}}</style>
</head><body><p>Đang đăng nhập…</p>
<script>
(function() {{
  var token = "{token}";
  var error = decodeURIComponent("{error}");
  if (error) {{
    sessionStorage.setItem("typoon_login_error", error);
  }} else if (token) {{
    localStorage.setItem("typoon_token", token);
  }}
  window.location.replace("{web_url}");
}})();
</script></body></html>"""
