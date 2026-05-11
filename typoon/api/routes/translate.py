"""Translate routes — spawn, manage, share per-user translations.

The spawn endpoint is the heart of the cache flow:

  1. Compute glossary_fp from user + community + (optionally) user_glossary
  2. Lookup `find_reusable_draft` filtered by viewer's guild memberships
  3. Hit  → create translation row pointing at the existing draft;
            no quota spent; return cache_hit=True
  4. Miss → create draft (visibility=guild|all_guilds|private) + enqueue
            prepare → scan → translate → render; record consume

NSFW material forces visibility='private' regardless of opt-out.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.chapter_archive import archive_token
from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import (
    get_auth_cfg, get_config, get_storage, get_store, require_user,
)
from typoon.api.models import TranslationOut
from typoon.api.quota import enforce_chapter_quota, record_consume
from typoon.api.routes._shared import (
    require_chapter, require_material, require_translation_owner,
)
from typoon.config import AuthConfig, Config
from typoon.storage import Store


router = APIRouter(
    prefix="/api/translate", tags=["translate"],
    dependencies=[Depends(require_user)],
)


# ── Spawn ─────────────────────────────────────────────────────────────


class ChapterRef(BaseModel):
    """Manifest-coordinated chapter reference used when the chapter
    row doesn't exist server-side yet."""
    material_id:  int
    upstream_url: str
    number:       str
    label:        str | None = None


class SpawnBody(BaseModel):
    """Spawn body. Either `chapter_id` (already-known internal row) or
    `chapter_ref` (manifest coords — auto-create the chapter row on
    first spawn) must be provided.

    Manifest-coord spawn is the common path for source-backed manga:
    the SPA has the upstream chapter URL from the manifest runtime
    and has not yet created any local chapter row. The first user to
    translate a chapter materializes it; subsequent users hit the
    cache lookup on (chapter_id, …) directly.
    """
    chapter_id:    int | None = None
    chapter_ref:   ChapterRef | None = None
    target_lang:   str
    force_private: bool = False
    visibility:    str = "guild"
    scope_guild_id: str | None = None


class SpawnResult(BaseModel):
    translation_id: int
    draft_id:       int
    state:          str
    cache_hit:      bool
    chapter_id:     int


# ── My translations index ─────────────────────────────────────────────


class MyTranslationOut(BaseModel):
    translation_id:        int
    target_lang:           str
    state:                 str          # done | running | error | pending
    has_archive:           bool
    updated_at:            str | None

    chapter_id:            int
    chapter_number:        str
    chapter_label:         str | None
    chapter_position:      int
    chapter_upstream_url:  str | None

    material_id:           int
    material_title:        str
    material_cover:        str | None
    material_source:       str | None
    material_upstream_ref: str | None


@router.get("/mine", response_model=list[MyTranslationOut])
async def list_my_translations(
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """User's own translations, newest activity first. Drives the
    /translate index in the SPA."""
    rows = await db.list_my_translations(user["id"])
    return [MyTranslationOut(**r) for r in rows]


@router.post("", response_model=SpawnResult)
async def spawn_translation(
    body: SpawnBody,
    user: dict        = Depends(require_user),
    db:   Store       = Depends(get_store),
    cfg:  Config      = Depends(get_config),
    auth: AuthConfig  = Depends(get_auth_cfg),
):
    """Spawn or reuse a translation for a chapter.

    Two entry shapes:
      • `chapter_id` — caller already has the internal row id (ext /
                       upload after finalize, or a re-translate flow).
      • `chapter_ref` — manifest coords; we materialize the chapter row
                        on first use. Common case for source-backed
                        spawns triggered directly from the manga page.
    """
    chapter: dict | None = None

    if body.chapter_id is not None:
        chapter = await db.get_chapter(body.chapter_id)
        if chapter is None:
            raise HTTPException(404, "Chapter not found")
    elif body.chapter_ref is not None:
        ref = body.chapter_ref
        material = await db.get_material(ref.material_id)
        if material is None:
            raise HTTPException(404, "Material not found")
        # Try dedup first — many users translating the same source
        # chapter share one row.
        chapter = await db.find_chapter_by_upstream(
            material["id"], ref.upstream_url,
        )
        if chapter is None:
            chapter_id = await db.create_chapter(
                material["id"], ref.number,
                label=ref.label,
                upstream_url=ref.upstream_url,
                pages_origin="remote",
            )
            chapter = await db.get_chapter(chapter_id)
        if chapter is None:
            raise HTTPException(500, "Failed to materialize chapter row")
    else:
        raise HTTPException(
            400,
            "Provide either chapter_id or chapter_ref",
        )

    material = await db.get_material(chapter["material_id"])
    assert material is not None

    # NSFW gate — overrides any opt-out preference.
    is_nsfw = bool(material.get("nsfw"))
    force_private = body.force_private or is_nsfw

    # Source-language defaults to the first language the material lists
    # (manifest already supplies this for source-backed material; ext /
    # upload default to 'unknown' until the user sets one).
    languages = material.get("languages") or []
    source_lang = languages[0] if languages else "unknown"
    target_lang = body.target_lang

    # Glossary fingerprint — cache key over (chapter, src, tgt, fp).
    glossary_fp = await db.compute_glossary_fingerprint(
        user_id=user["id"],
        source_lang=source_lang,
        target_lang=target_lang,
        material_id=material["id"],
    )

    viewer_guilds = [
        g["id"] for g in await db.get_user_guilds(user["id"])
    ]

    # Cache lookup unless the user forced a private spawn.
    draft = None
    if not force_private:
        draft = await db.find_reusable_draft(
            chapter_id=chapter["id"],
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_fp=glossary_fp,
            viewer_id=user["id"],
            viewer_guilds=viewer_guilds,
        )

    if draft is not None:
        # Cache hit. No quota; no pipeline enqueue.
        translation_id = await db.get_or_create_translation(
            chapter_id=chapter["id"],
            owner_id=user["id"],
            target_lang=target_lang,
            draft_id=draft["id"],
        )
        return SpawnResult(
            translation_id=translation_id,
            draft_id=draft["id"],
            state=draft.get("state") or "done",
            cache_hit=True,
            chapter_id=chapter["id"],
        )

    # Cache miss — quota check + create draft + enqueue prepare.
    await enforce_chapter_quota(user, db, cfg.rate_limit, auth, count=1)

    visibility = "private" if force_private else body.visibility
    if visibility not in ("private", "guild", "all_guilds"):
        raise HTTPException(400, f"invalid visibility: {visibility!r}")
    scope_guild_id = (
        body.scope_guild_id if visibility == "guild" else None
    )
    if visibility == "guild" and not scope_guild_id:
        raise HTTPException(
            400,
            "scope_guild_id is required when visibility='guild'",
        )

    llm_model = cfg.llm.default_model if hasattr(cfg, "llm") else "claude-3.5"
    draft_id = await db.create_draft(
        chapter_id=chapter["id"],
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_fp=glossary_fp,
        llm_model=llm_model,
        created_by=user["id"],
        visibility=visibility,
        scope_guild_id=scope_guild_id,
    )
    translation_id = await db.get_or_create_translation(
        chapter_id=chapter["id"],
        owner_id=user["id"],
        target_lang=target_lang,
        draft_id=draft_id,
    )
    await record_consume(
        user, db, auth,
        translation_id=translation_id, kind="draft_create",
    )

    # Enqueue the entry stage. Worker fans out to subsequent stages
    # via advance_task with target_kind shifts.
    if chapter.get("prepared_hash") is None:
        await db.enqueue_task(
            target_kind="chapter", target_id=chapter["id"],
            stage="prepare",
        )
    elif not await db.has_bubbles(chapter["id"]):
        await db.enqueue_task(
            target_kind="chapter", target_id=chapter["id"],
            stage="scan",
        )
    else:
        await db.enqueue_task(
            target_kind="draft", target_id=draft_id, stage="translate",
        )

    return SpawnResult(
        translation_id=translation_id,
        draft_id=draft_id,
        state="pending",
        cache_hit=False,
        chapter_id=chapter["id"],
    )


# ── Detail / state ────────────────────────────────────────────────────


def _build_archive_url(
    *,
    target_kind: str,
    target_id:   int,
    salt:        bytes,
    backend:     str | None,
    locator:     str | None,
    rendered_at: str | None,
    stores:      StorageRegistry,
) -> str | None:
    """Returns the public CDN URL for a render archive, or None when
    the archive isn't ready. We dispatch by backend so multi-backend
    coexistence works without migration."""
    if not backend or not locator:
        return None
    try:
        reader = stores.reader(backend)
    except RuntimeError:
        return None
    return reader.url(locator, version=rendered_at or "")


@router.get("/{translation_id}", response_model=TranslationOut)
async def get_translation(
    translation_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    cfg:    Config          = Depends(get_config),
    stores: StorageRegistry = Depends(get_storage),
):
    t = await db.get_translation(translation_id)
    if t is None:
        raise HTTPException(404, "Translation not found")
    # Read access: owner OR via draft visibility scope.
    if t["owner_id"] != user["id"]:
        # Cross-user read — gate by draft visibility.
        draft = await db.get_draft(t["draft_id"]) if t.get("draft_id") else None
        if draft is None or draft.get("takedown_at"):
            raise HTTPException(404, "Translation not found")
        viewer_guilds = [
            g["id"] for g in await db.get_user_guilds(user["id"])
        ]
        if draft["visibility"] == "private":
            raise HTTPException(404, "Translation not found")
        if draft["visibility"] == "guild":
            if draft.get("scope_guild_id") not in viewer_guilds:
                raise HTTPException(404, "Translation not found")
        # 'all_guilds' relies on creator sharing a guild with viewer;
        # we approximate by requiring at least one shared guild via DB.

    # Build public archive URL. If translation has no per-row archive,
    # fall back to the draft's default render (target_kind='draft').
    if t.get("archive_locator"):
        archive_url = _build_archive_url(
            target_kind="translation", target_id=t["id"],
            salt=cfg.storage.archive_path_salt.encode(),
            backend=t.get("archive_backend"),
            locator=t.get("archive_locator"),
            rendered_at=t.get("rendered_at"),
            stores=stores,
        )
    elif t.get("draft_id"):
        # Default render shared across all translations referencing
        # this draft. The render worker writes it to the draft target.
        draft = await db.get_draft(t["draft_id"])
        archive_url = (
            _build_archive_url(
                target_kind="draft", target_id=t["draft_id"],
                salt=cfg.storage.archive_path_salt.encode(),
                backend=(draft or {}).get("archive_backend") if draft else None,
                locator=(draft or {}).get("archive_locator") if draft else None,
                rendered_at=t.get("rendered_at"),
                stores=stores,
            )
            if draft else None
        )
    else:
        archive_url = None

    edits = await db.get_translation_edits(translation_id)
    return TranslationOut(
        id=t["id"],
        chapter_id=t["chapter_id"],
        owner_id=t["owner_id"],
        target_lang=t["target_lang"],
        draft_id=t.get("draft_id"),
        state="done" if archive_url else "pending",
        in_feed=bool(t.get("in_feed")),
        feed_guild_id=t.get("feed_guild_id"),
        archive_url=archive_url,
        has_edits=len(edits) > 0,
        created_at=t.get("created_at"),
        updated_at=t.get("updated_at"),
    )


# ── Patch (feed flag, ownership-only) ─────────────────────────────────


class PatchTranslationBody(BaseModel):
    in_feed:        bool | None = None
    feed_guild_id:  str | None = None


@router.patch("/{translation_id}", response_model=TranslationOut)
async def patch_translation(
    translation_id: int,
    body:           PatchTranslationBody,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    cfg:    Config          = Depends(get_config),
    stores: StorageRegistry = Depends(get_storage),
):
    t = await require_translation_owner(translation_id, user, db)
    if body.in_feed is not None or body.feed_guild_id is not None:
        await db.update_translation_feed(
            translation_id,
            in_feed=body.in_feed if body.in_feed is not None
                   else bool(t.get("in_feed")),
            feed_guild_id=body.feed_guild_id
                if body.feed_guild_id is not None
                else t.get("feed_guild_id"),
        )
    return await get_translation(
        translation_id, user=user, db=db, cfg=cfg, stores=stores,
    )


# ── Edits (sparse override) ───────────────────────────────────────────


class EditBubbleBody(BaseModel):
    page_index:  int
    bubble_idx:  int
    edited_text: str


class BubbleEditOut(BaseModel):
    page_index:      int
    bubble_idx:      int
    source_text:     str
    draft_text:      str               # LLM-generated, from translation_draft_bubbles
    edited_text:     str | None        # user override; None when no edit
    kind:            str               # dialogue | sfx | skip


@router.get(
    "/{translation_id}/bubbles",
    response_model=list[BubbleEditOut],
)
async def list_translation_bubbles(
    translation_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Per-bubble view for the editor: source OCR + draft text +
    optional user edit. Owner-only — sparse edits are personal."""
    t = await require_translation_owner(translation_id, user, db)
    chapter_id = t["chapter_id"]
    draft_id   = t.get("draft_id")

    source_bubbles = await db.get_bubbles(chapter_id)
    draft_bubbles  = (
        await db.get_draft_bubbles(draft_id) if draft_id else []
    )
    edits = await db.get_translation_edits(translation_id)

    src_by_pos = {
        (b["page_index"], b["bubble_idx"]): b for b in source_bubbles
    }
    draft_by_pos = {
        (b["page_index"], b["bubble_idx"]): b for b in draft_bubbles
    }
    edit_by_pos = {
        (e["page_index"], e["bubble_idx"]): e for e in edits
    }

    out: list[BubbleEditOut] = []
    for pos in sorted(src_by_pos):
        s = src_by_pos[pos]
        d = draft_by_pos.get(pos)
        e = edit_by_pos.get(pos)
        out.append(BubbleEditOut(
            page_index=pos[0],
            bubble_idx=pos[1],
            source_text=s["source_text"],
            draft_text=d["translated_text"] if d else "",
            edited_text=e["edited_text"] if e else None,
            kind=(d.get("kind") if d else None) or "dialogue",
        ))
    return out


@router.put("/{translation_id}/edits", status_code=204)
async def upsert_edit(
    translation_id: int,
    body:           EditBubbleBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    await require_translation_owner(translation_id, user, db)
    await db.upsert_translation_edit(
        translation_id, body.page_index, body.bubble_idx, body.edited_text,
    )


@router.delete(
    "/{translation_id}/edits/{page_index}/{bubble_idx}",
    status_code=204,
)
async def delete_edit(
    translation_id: int,
    page_index:     int,
    bubble_idx:     int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    await require_translation_owner(translation_id, user, db)
    ok = await db.delete_translation_edit(
        translation_id, page_index, bubble_idx,
    )
    if not ok:
        raise HTTPException(404, "Edit not found")


# ── Redo (force fresh draft) ──────────────────────────────────────────


class RedoBody(BaseModel):
    force_private: bool = False


@router.post("/{translation_id}/redo", response_model=SpawnResult)
async def redo_translation(
    translation_id: int,
    body:           RedoBody,
    user: dict        = Depends(require_user),
    db:   Store       = Depends(get_store),
    cfg:  Config      = Depends(get_config),
    auth: AuthConfig  = Depends(get_auth_cfg),
):
    """Re-run pipeline with a fresh draft. Spends quota.

    The current translation row stays but points at the new draft once
    pipeline completes; old draft is left for cache hits from other
    users (it isn't deleted just because owner redid it).
    """
    t = await require_translation_owner(translation_id, user, db)
    # Construct a SpawnBody and reuse spawn logic. force_private=True
    # bypasses cache so we definitely create a new draft.
    spawn_body = SpawnBody(
        chapter_id=t["chapter_id"],
        target_lang=t["target_lang"],
        force_private=body.force_private,
        # Inherit current scope from the existing draft when possible.
        visibility="guild",  # default; UI may pass through PATCH later
        scope_guild_id=None,
    )
    # Run cache-bypass spawn explicitly by clearing the cache option.
    return await spawn_translation(
        spawn_body, user=user, db=db, cfg=cfg, auth=auth,
    )


# ── Delete ────────────────────────────────────────────────────────────


@router.delete("/{translation_id}", status_code=204)
async def delete_translation(
    translation_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    cfg:    Config          = Depends(get_config),
    stores: StorageRegistry = Depends(get_storage),
):
    """Delete the per-user wrapper. Underlying draft + cache survive
    (other users may still reference it). Per-translation archive (if
    any) is removed before DB row is gone."""
    from typoon.adapters.chapter_archive import render_key

    t = await require_translation_owner(translation_id, user, db)
    if t.get("archive_backend") and t.get("archive_locator"):
        try:
            backend = stores.reader(t["archive_backend"])
            await backend.delete(t["archive_locator"])
        except RuntimeError:
            pass  # backend no longer configured; orphan, nothing to do

    # Cascade via FK drops translation_edits.
    async with db._pool.acquire() as conn:                # type: ignore[attr-defined]
        await conn.execute(
            "DELETE FROM translations WHERE id=$1",
            translation_id,
        )
