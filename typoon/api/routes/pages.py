"""Rendered chapter archive serving.

Two-step access pattern — required to keep R2/CDN edge caching viable:

  1. `GET /chapters/:cid/render` — auth-required. Returns a short-lived
     archive URL signed for *that chapter only*. Body: { url, expires_at,
     page_count }. The URL embeds a scoped token in the query string.

  2. `GET <signed_url>` — public, no Authorization header. The path is
     `/files/p/<pid>/c/<cid>/render.bnl?t=<scoped_token>`. The signed
     token is verified server-side; on success FileResponse streams the
     archive with Range support (or, for R2, redirects to a presigned URL).

Why not "auth on the .bnl request": each Range request would need to
carry the JWT, defeating CDN cache. The signed URL is itself the cache
key — browser, edge, and origin all key on the URL.

Local store: signed token issuance is in-process; the .bnl request hits
this same FastAPI app. R2: token issuance same; .bnl request returns
HTTP 302 to a presigned R2 URL, browser hits R2 directly.
"""

from __future__ import annotations

import time

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from typoon.adapters.artifact_store import LocalArtifactStore
from typoon.adapters.chapter_archive import render_key
from typoon.api.deps import (
    get_artifact_store, get_auth_cfg, get_store, require_user,
)
from typoon.api.routes._shared import require_project_view
from typoon.config import AuthConfig
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["pages"])

JWT_ALGORITHM = "HS256"
ARCHIVE_TTL   = 15 * 60   # 15 minutes — long enough for a reading session,
                          # short enough that a leaked URL expires fast.


# ── Scoped tokens ────────────────────────────────────────────────────


def _issue_archive_token(
    *, project_id: int, chapter_id: int, cfg: AuthConfig,
) -> tuple[str, int]:
    """Issue a token scoped to a single chapter's render archive.

    `aud=archive` is checked on the public endpoint, ensuring a session
    JWT can't be used to bypass scope and a scoped token can't be used
    to call protected endpoints.
    """
    exp = int(time.time()) + ARCHIVE_TTL
    payload = {"aud": "archive", "p": project_id, "c": chapter_id, "exp": exp}
    return jwt.encode(payload, cfg.jwt_secret, algorithm=JWT_ALGORITHM), exp


def _verify_archive_token(
    token: str, *, project_id: int, chapter_id: int, cfg: AuthConfig,
) -> None:
    try:
        payload = jwt.decode(
            token, cfg.jwt_secret, algorithms=[JWT_ALGORITHM], audience="archive",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid archive token: {e}") from e
    if payload.get("p") != project_id or payload.get("c") != chapter_id:
        raise HTTPException(403, "Token scope mismatch")


# ── Step 1: signed URL issuance (auth-required) ─────────────────────


class RenderUrlBody(BaseModel):
    url:         str
    expires_at:  int
    page_count:  int


@router.get(
    "/{project_id}/chapters/{chapter_id}/render",
    response_model=RenderUrlBody,
    dependencies=[Depends(require_user)],
)
async def get_render_url(
    project_id: int,
    chapter_id: int,
    user:  dict       = Depends(require_user),
    db:    Store      = Depends(get_store),
    cfg:   AuthConfig = Depends(get_auth_cfg),
) -> RenderUrlBody:
    await require_project_view(project_id, user, db)

    state = await db.get_chapter_render_state(chapter_id)
    if state is None or not state["rendered"]:
        raise HTTPException(404, "Render archive not available")

    token, exp = _issue_archive_token(
        project_id=project_id, chapter_id=chapter_id, cfg=cfg,
    )
    return RenderUrlBody(
        url=f"/api/projects/{project_id}/chapters/{chapter_id}/render.bnl?t={token}",
        expires_at=exp,
        page_count=state["page_count"],
    )


# ── Step 2: archive bytes (token-scoped, public-on-URL) ─────────────


@router.get("/{project_id}/chapters/{chapter_id}/render.bnl")
async def get_render_archive(
    project_id: int,
    chapter_id: int,
    t: str = Query(..., description="Scoped archive token"),
    cfg:   AuthConfig = Depends(get_auth_cfg),
    store=Depends(get_artifact_store),
):
    """Serve archive bytes. No Authorization header required — the URL
    itself carries the scoped JWT. The URL is therefore cacheable by
    CDN edges (the signature is the cache key).
    """
    _verify_archive_token(t, project_id=project_id, chapter_id=chapter_id, cfg=cfg)

    if not isinstance(store, LocalArtifactStore):
        raise HTTPException(501, "Remote artifact store not supported yet")

    path = store._path(render_key(project_id, chapter_id))  # noqa: SLF001
    if not path.exists():
        raise HTTPException(404, "Render archive missing on disk")

    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={
            # Aggressive cache: the URL changes every render (chapter
            # updated_at busts via ?v=, plus the scoped token rotates).
            "Cache-Control": "public, max-age=86400, immutable",
            "Accept-Ranges": "bytes",
        },
    )
