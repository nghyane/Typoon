"""Translate routes — spawn, manage, share per-user translations.

The spawn endpoint is the heart of the cache flow:

  1. Compute glossary_fp from user + community + (optionally) user_glossary
  2. Lookup `find_reusable_draft` — schema 19 made the cache pool
     global, so any matching draft is reusable
  3. Hit  → create translation row pointing at the existing draft;
            no quota spent; return cache_hit=True
  4. Miss → create draft + enqueue prepare → scan → translate → render;
            record consume
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import (
    get_auth_cfg, get_config, get_storage, get_store, require_user,
)
from typoon.api.models import TranslationOut
from typoon.api.quota import enforce_chapter_quota, record_consume
from typoon.api.routes._shared import (
    require_translation_owner,
    resolve_archive_url,
)
from typoon.config import AuthConfig, Config
from typoon.storage import Store


router = APIRouter(
    prefix="/api/translate", tags=["translate"],
    dependencies=[Depends(require_user)],
)


class SpawnBody(BaseModel):
    """Spawn body. `chapter_id` must refer to an existing chapter row.
    Use the upload finalize endpoint to create a chapter row first.
    """
    chapter_id:  int
    target_lang: str


class SpawnResult(BaseModel):
    translation_id: int
    draft_id:       int
    state:          str
    cache_hit:      bool
    chapter_id:     int


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

    `chapter_id` must refer to an existing chapter row. Use the upload
    finalize endpoint to create a chapter row before calling this.
    """
    chapter = await db.get_chapter(body.chapter_id)
    if chapter is None:
        raise HTTPException(404, "Chapter not found")

    material = await db.get_material(chapter["material_id"])
    assert material is not None

    # Source-language source of truth is the chapter row, populated
    # at upload-finalize from the client's `version.lang`. Falls back
    # to the material's primary language for legacy rows created
    # before schema 26 (the backfill UPDATE seeded those from the
    # same source, but new uploads must carry it explicitly).
    chapter_lang = (chapter.get("source_lang") or "").strip().lower() or None
    if chapter_lang:
        source_lang = chapter_lang
    else:
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

    # Cache lookup against the global community pool.
    draft = await db.find_reusable_draft(
        chapter_id=chapter["id"],
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_fp=glossary_fp,
    )

    if draft is not None:
        # Lookup hit. The state of the existing draft decides whether
        # we serve as cache, piggy-back on an in-flight run, or auto-
        # restart a prior failure. Either way the user pays no quota
        # for an existing key — that was charged at first spawn.
        translation_id = await db.get_or_create_translation(
            work_chapter_id=chapter["work_chapter_id"],
            owner_id=user["id"],
            target_lang=target_lang,
            draft_id=draft["id"],
            shared=not bool(material.get("nsfw")),
        )
        state = draft.get("state") or "pending"
        if state == "error":
            # Auto-redo on spawn. The previous owner left a broken
            # draft on this exact cache key; the new spawner expects
            # a working pipeline, not an inherited error. Restarting
            # is the only way to honor that without forcing them to
            # discover the error and click redo themselves.
            stage, kind, tid = await _resolve_entry_task(
                db, chapter, int(draft["id"]),
            )
            # Race: if the draft was taken down between find and
            # restart, the bool comes back False — we fall through
            # to the cache-miss path below by re-running the lookup.
            # Practically negligible; ignore.
            await db.restart_draft_pipeline(
                int(draft["id"]),
                entry_stage=stage,
                entry_target_kind=kind,
                entry_target_id=tid,
            )
            state = "pending"
        return SpawnResult(
            translation_id=translation_id,
            draft_id=draft["id"],
            state=state,
            cache_hit=(state == "done"),
            chapter_id=chapter["id"],
        )

    # Cache miss — quota check + create draft + enqueue prepare.
    await enforce_chapter_quota(user, db, cfg.rate_limit, auth, count=1)

    # Cache key dimension — the draft is keyed on (chapter, src, tgt,
    # glossary_fp, llm_model), so the model string must come from
    # config rather than a fallback string that drifts silently when
    # the actual LLM is upgraded.
    llm_model = cfg.translation.model
    draft_id = await db.create_draft(
        chapter_id=chapter["id"],
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_fp=glossary_fp,
        llm_model=llm_model,
        created_by=user["id"],
    )
    translation_id = await db.get_or_create_translation(
        work_chapter_id=chapter["work_chapter_id"],
        owner_id=user["id"],
        target_lang=target_lang,
        draft_id=draft_id,
        shared=not bool(material.get("nsfw")),
    )
    await record_consume(
        user, db, auth,
        translation_id=translation_id, kind="draft_create",
    )

    # Enqueue the entry stage. Worker fans out to subsequent stages
    # via advance_task with target_kind shifts.
    stage, kind, tid = await _resolve_entry_task(db, chapter, draft_id)
    await db.enqueue_task(target_kind=kind, target_id=tid, stage=stage)

    return SpawnResult(
        translation_id=translation_id,
        draft_id=draft_id,
        state="pending",
        cache_hit=False,
        chapter_id=chapter["id"],
    )


async def _serialize_translation(
    t:      dict,
    *,
    db:     Store,
    stores: StorageRegistry,
) -> TranslationOut:
    """Compose the full TranslationOut from a translation row.

    Pulls the archive URL (with draft-default fallback), the per-row
    edit count, and the chapter/material denormalisation that the SPA
    needs to render a card without a second round-trip. Routes use
    this so the serialisation rule is owned in one place — patch
    handlers don't re-enter the GET route just to rebuild a response.
    """
    archive_url = await resolve_archive_url(t, db=db, stores=stores)
    edits = await db.get_translation_edits(t["id"])
    # Translations live at Work-chapter scope; the representative
    # per-source chapter is reached via the draft pointer (the pixel
    # the translation actually serves).
    draft = (
        await db.get_draft(t["draft_id"]) if t.get("draft_id") else None
    )
    chapter = (
        await db.get_chapter(draft["chapter_id"]) if draft else None
    )
    material = (
        await db.get_material(chapter["material_id"]) if chapter else None
    )
    # work_id from material; work_chapter_id from chapter (joined into
    # `get_chapter`/`list_chapters` payloads alongside number_norm).
    work_id = int((material or {}).get("work_id") or 0)
    work_chapter_id = int(t.get("work_chapter_id") or 0)
    return TranslationOut(
        id=t["id"],
        work_id=work_id,
        work_chapter_id=work_chapter_id,
        chapter_id=(chapter or {}).get("id") or 0,
        material_id=(chapter or {}).get("material_id") or 0,
        owner_id=t["owner_id"],
        target_lang=t["target_lang"],
        draft_id=t.get("draft_id"),
        state="done" if archive_url else "pending",
        archive_url=archive_url,
        has_edits=len(edits) > 0,
        chapter_number=(chapter or {}).get("number_norm"),
        chapter_label=(chapter or {}).get("label"),
        material_title=(material or {}).get("title"),
        shared=bool(t.get("shared", True)),
        created_at=t.get("created_at"),
        updated_at=t.get("updated_at"),
    )


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
    if t.get("takedown_at"):
        # DMCA / moderator takedown — invisible to everyone.
        raise HTTPException(404, "Translation not found")
    # Schema 19: every alive translation is community-readable. The
    # only gate left is "is this translation taken down?" — anyone
    # logged in can open any non-taken-down translation.
    return await _serialize_translation(t, db=db, stores=stores)


class PatchTranslationBody(BaseModel):
    """Patch surface left for forward-compat (rename, future flags).

    Schema 19 dropped the `in_feed` / `feed_guild_id` toggles —
    every translation is community-visible by default; to hide one,
    the owner deletes it.
    """
    # Reserved for future fields; currently no-op patch is allowed
    # so the SPA can keep its existing PATCH plumbing.


@router.patch("/{translation_id}", response_model=TranslationOut)
async def patch_translation(
    translation_id: int,
    body:           PatchTranslationBody,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    del body  # no patchable fields today; kept for API stability
    t = await require_translation_owner(translation_id, user, db)
    return await _serialize_translation(t, db=db, stores=stores)


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
    draft_id = t["draft_id"]
    # Pixel-bound bubbles live on the draft's chapter, not the
    # translation (which sits at Work-chapter scope).
    draft = await db.get_draft(draft_id)
    if draft is None:
        raise HTTPException(500, "Translation draft missing")
    chapter_id = draft["chapter_id"]

    source_bubbles = await db.get_bubbles(chapter_id)
    draft_bubbles  = await db.get_draft_bubbles(draft_id)
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


async def _resolve_entry_task(
    db:        Store,
    chapter:   dict,
    draft_id:  int,
) -> tuple[str, str, int]:
    """Derive (stage, target_kind, target_id) for the entry task of
    a draft's pipeline.

    The entry point depends on how far the chapter's *shared* state
    has progressed:

      - no prepared pixels   → start at chapter prepare
      - prepared but no OCR  → start at chapter scan
      - both done            → start at draft translate

    Shared by spawn (cache miss path) and redo so the two never
    disagree on where to enter.
    """
    if chapter.get("prepared_hash") is None:
        return ("prepare", "chapter", int(chapter["id"]))
    if not await db.has_bubbles(int(chapter["id"])):
        return ("scan", "chapter", int(chapter["id"]))
    return ("translate", "draft", int(draft_id))


@router.post("/{translation_id}/redo", response_model=SpawnResult)
async def redo_translation(
    translation_id: int,
    user: dict   = Depends(require_user),
    db:   Store  = Depends(get_store),
):
    """Re-run the pipeline for an owned translation.

    The draft's current state decides what redo means:

      - `done`              → nothing to do; the output is canonical.
                              Returns cache_hit=True, no work queued.
      - `error`             → reset draft + entry task atomically and
                              re-enqueue. The user already paid quota
                              for this (chapter, src, tgt, glossary)
                              key when they spawned, so retry is free.
      - `pending`/`running` → already in flight; nothing to do.
                              Returns the live state so the SPA keeps
                              polling rather than thrashing the queue.
      - `blocked`           → policy hold; redo is not the right tool.
                              Returns 409 so the UI surfaces the
                              moderation state instead of pretending
                              to retry.

    For admin force-restart of a `done` or `blocked` draft (e.g. a
    model upgrade invalidates an output, or moderation review
    cleared a hold), see `POST /api/admin/ops/drafts/{draft_id}/restart`
    which works regardless of state and writes an audit row.
    """
    t = await require_translation_owner(translation_id, user, db)
    draft = await db.get_draft(t["draft_id"])
    if draft is None:
        raise HTTPException(500, "Translation draft missing")

    state = draft.get("state") or "pending"
    chapter_id = int(draft["chapter_id"])

    if state == "done":
        return SpawnResult(
            translation_id=translation_id,
            draft_id=int(draft["id"]),
            state="done",
            cache_hit=True,
            chapter_id=chapter_id,
        )
    if state in ("pending", "running"):
        return SpawnResult(
            translation_id=translation_id,
            draft_id=int(draft["id"]),
            state=state,
            cache_hit=True,
            chapter_id=chapter_id,
        )
    if state == "blocked":
        raise HTTPException(
            409, "Draft is moderation-blocked; redo is not allowed",
        )

    # state == 'error' (or any future terminal-failure state).
    chapter = await db.get_chapter(chapter_id)
    if chapter is None:
        raise HTTPException(500, "Draft references missing chapter")
    stage, kind, tid = await _resolve_entry_task(
        db, chapter, int(draft["id"]),
    )
    ok = await db.restart_draft_pipeline(
        int(draft["id"]),
        entry_stage=stage,
        entry_target_kind=kind,
        entry_target_id=tid,
    )
    if not ok:
        # Draft vanished between get_draft and the FOR UPDATE — treat
        # as 404 so the SPA refetches and renders the empty state.
        raise HTTPException(404, "Translation draft missing")
    return SpawnResult(
        translation_id=translation_id,
        draft_id=int(draft["id"]),
        state="pending",
        cache_hit=False,
        chapter_id=chapter_id,
    )


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
    t = await require_translation_owner(translation_id, user, db)
    if t.get("archive_backend") and t.get("archive_locator"):
        try:
            backend = stores.reader(t["archive_backend"])
            await backend.delete(t["archive_locator"])
        except RuntimeError:
            pass  # backend no longer configured; orphan, nothing to do

    # Cascade via FK drops translation_edits.
    await db.delete_translation(translation_id)
