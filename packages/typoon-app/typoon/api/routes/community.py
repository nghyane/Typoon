"""Community-wide recent translations.

Schema 19 removed Discord-guild scoping — the community is a single
global pool. `/api/community/recent` lists the most-recent translation
per material across every user; the home page renders it as a
"Mới trong cộng đồng" section.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_storage, get_store, require_user
from typoon.api.models import CommunityFeedEntryOut
from typoon.api.routes._shared import resolve_archive_url
from typoon.storage.store import Store

router = APIRouter(
    prefix="/api/community", tags=["community"],
    dependencies=[Depends(require_user)],
)


@router.get("/recent", response_model=list[CommunityFeedEntryOut])
async def list_recent(
    before: str | None = Query(None),
    limit:  int        = Query(60, ge=1, le=200),
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    """Recent translations across the community, deduped per material.

    Cursor pagination via ISO timestamp `before` — pass the per-material
    `created_at` of the last surfaced row to fetch the next page.
    """
    rows = await db.list_recent_community(
        viewer_id=user["id"], limit=limit, before=before,
    )

    out: list[CommunityFeedEntryOut] = []
    for r in rows:
        # Resolve archive URL via the translation row (and its draft
        # fallback). At ≤60 rows per page this is cheap; if it becomes
        # hot, push the join into Store.
        t = await db.get_translation(r["translation_id"])
        archive_url = (
            await resolve_archive_url(t, db=db, stores=stores)
            if t else None
        )

        out.append(CommunityFeedEntryOut(
            translation_id=r["translation_id"],
            chapter_id=r["chapter_id"],
            chapter_number=r["chapter_number"],
            chapter_label=r.get("chapter_label"),
            work_id=int(r["work_id"]),
            material_id=r["material_id"],
            title=r["title"],
            cover=r.get("cover"),
            target_lang=r["target_lang"],
            creator_id=r.get("creator_id"),
            creator_name=r.get("creator_name"),
            created_at=r.get("created_at"),
            archive_url=archive_url,
            chapters_in_feed=int(r.get("chapters_in_feed") or 1),
        ))
    return out
