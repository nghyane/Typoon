"""User self-service endpoints — identity, API tokens, quota,
reading history.

Schema 19 removed Discord guild scoping — `/me` no longer carries a
guild list because the community is a single global pool.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from typoon.api.auth_token import issue_api_token
from typoon.api.deps import get_auth_cfg, get_config, get_store, require_user
from typoon.api.models import MeOut, RecentReadOut
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


class RecordTranslatedReadingBody(BaseModel):
    """Reader opened a translation. The translation already pins a
    Work chapter and a representative material (via translation_drafts
    → chapters), so we only need the translation_id.
    """
    translation_id: int


class RecordRawReadingBody(BaseModel):
    """Reader opened a raw chapter. No DB chapter row may exist yet —
    we materialise the Work chapter on demand using the material id
    plus the manifest-supplied normalised number so the history entry
    can dedup across sources of the same Work.
    """
    material_id: int
    number:      str
    number_norm: str
    label:       str | None = None


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
async def me(user: dict = Depends(require_user)):
    """Current user identity. Slim — no guild list, no settings
    payload; clients ask `/api/me/quota` etc. as needed."""
    return MeOut(
        id=user["id"],
        display_name=user["display_name"],
        avatar_url=user.get("avatar_url"),
    )


# ── Reading history ──────────────────────────────────────────────────


@router.get("/recent-reads", response_model=list[RecentReadOut])
async def list_recent_reads(
    user:  dict       = Depends(require_user),
    db:    Store      = Depends(get_store),
    limit: int        = 30,
):
    """Recently-read manga, newest first. Drives the home
    "Tiếp tục đọc" surface."""
    rows = await db.list_recent_reads(user_id=user["id"], limit=limit)
    return [RecentReadOut(**r) for r in rows]


@router.post("/reading/translated", status_code=204)
async def record_translated_reading(
    body: RecordTranslatedReadingBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Record a translated read. Resolves the (work_chapter, material)
    pair via the translation's draft → chapter so the history row can
    dedupe across sources of the same Work.
    """
    trans = await db.get_translation(body.translation_id)
    if trans is None:
        raise HTTPException(404, "Translation not found")
    draft = await db.get_draft(trans["draft_id"])
    if draft is None:
        raise HTTPException(500, "Translation draft missing")
    chapter = await db.get_chapter(draft["chapter_id"])
    if chapter is None:
        raise HTTPException(500, "Draft chapter missing")
    await db.record_reading(
        user_id=user["id"],
        work_chapter_id=int(trans["work_chapter_id"]),
        last_material_id=int(chapter["material_id"]),
        translation_id=body.translation_id,
    )


@router.post("/reading/raw", status_code=204)
async def record_raw_reading(
    body: RecordRawReadingBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Record a raw read. Materialises the Work chapter on demand so
    history entries collapse per (user, Work chapter) regardless of
    which source the user opened. The caller-supplied `number_norm`
    is the manifest runtime's declarative normalisation output.
    """
    material = await db.get_material(body.material_id)
    if material is None:
        raise HTTPException(404, "Material not found")
    work_chapter_id = await db.find_or_create_work_chapter(
        work_id=int(material["work_id"]),
        number_norm=body.number_norm,
        label=body.label,
    )
    await db.record_reading(
        user_id=user["id"],
        work_chapter_id=work_chapter_id,
        last_material_id=body.material_id,
        translation_id=None,
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
