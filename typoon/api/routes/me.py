"""User self-service endpoints — RFC-008 API tokens."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from typoon.api.auth_token import issue_api_token
from typoon.api.deps import get_store, require_user
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


# ── Routes ────────────────────────────────────────────────────────────


@router.get("/projects")
async def my_projects(
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Convenience alias of /api/projects?filter=mine.

    Tools (extension, CLI) call this without needing to know about
    filter query params.
    """
    rows = await db.list_projects(viewer_id=user["id"], filter="mine")
    return [
        {
            "project_id":  r["id"],
            "slug":        r["slug"],
            "title":       r["title"],
            "cover_url":   f"/files/{r['slug']}/cover.jpg" if r.get("cover_path") else None,
            "source_lang": r["source_lang"],
            "target_lang": r["target_lang"],
            "shared":      bool(r.get("shared")),
        }
        for r in rows
    ]


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
    """Create a new API token. The plaintext is in the response and
    will never be retrievable again — caller must show + persist it
    immediately.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    token_id, plaintext, prefix = await issue_api_token(
        db, user_id=user["id"], name=name,
    )
    # Round-trip through list_api_tokens to get the same shape (with
    # null last_used + created_at as ISO strings).
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
