"""Feed routes — Hội Mê Truyện, guild-scoped.

`/api/feed/guild/{id}` lists translations with `in_feed=TRUE` scoped
to the guild. Viewer must be a member of the guild (membership check
against user_guilds cache).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import (
    get_config, get_storage, get_store, require_user,
)
from typoon.api.models import FeedEntryOut
from typoon.api.routes.translate import _build_archive_url
from typoon.config import Config
from typoon.storage import Store


router = APIRouter(
    prefix="/api/feed", tags=["feed"],
    dependencies=[Depends(require_user)],
)


@router.get("/guild/{guild_id}", response_model=list[FeedEntryOut])
async def list_guild_feed(
    guild_id: str,
    before:   str | None = Query(None),
    limit:    int        = Query(50, ge=1, le=200),
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    cfg:    Config          = Depends(get_config),
    stores: StorageRegistry = Depends(get_storage),
):
    """List feed entries for a guild. Member-only.

    Cursor pagination via ISO timestamp `before` — pass the last seen
    `created_at` to fetch the next page.
    """
    if not await db.user_in_guild(user["id"], guild_id):
        # 404 over 403 — hide existence of guilds the user isn't in.
        raise HTTPException(404, "Guild not found")

    rows = await db.list_feed_entries(
        guild_id=guild_id, viewer_id=user["id"],
        limit=limit, before=before,
    )

    out: list[FeedEntryOut] = []
    for r in rows:
        # Build archive URL via the draft pointer (shared default
        # render) or fall back to the translation row.
        archive_url = None
        # The list_feed_entries query doesn't join draft fields for
        # render dispatch; we look up archive backend lazily per row.
        # In practice the feed is paginated to ≤50 rows so this is
        # cheap. If it becomes hot, push the join into Store.
        t = await db.get_translation(r["translation_id"])
        if t and t.get("archive_locator"):
            archive_url = _build_archive_url(
                target_kind="translation", target_id=t["id"],
                salt=cfg.storage.archive_path_salt.encode(),
                backend=t.get("archive_backend"),
                locator=t.get("archive_locator"),
                rendered_at=t.get("rendered_at"),
                stores=stores,
            )
        elif t and t.get("draft_id"):
            d = await db.get_draft(t["draft_id"])
            if d:
                archive_url = _build_archive_url(
                    target_kind="draft", target_id=d["id"],
                    salt=cfg.storage.archive_path_salt.encode(),
                    backend=d.get("archive_backend"),
                    locator=d.get("archive_locator"),
                    rendered_at=t.get("rendered_at"),
                    stores=stores,
                )

        out.append(FeedEntryOut(
            translation_id=r["translation_id"],
            chapter_id=r["chapter_id"],
            chapter_number=r["chapter_number"],
            chapter_label=r.get("chapter_label"),
            material_id=r["material_id"],
            material_title=r["material_title"],
            material_cover=r.get("material_cover"),
            target_lang=r["target_lang"],
            creator_id=r.get("creator_id"),
            creator_name=r.get("creator_name"),
            created_at=r.get("created_at"),
            archive_url=archive_url,
        ))
    return out
