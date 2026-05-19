"""Work routes — global identity hub for cross-source manga pages.

A Work groups every sibling material (per source) under one identity.
The SPA's canonical MangaPage lives at `/w/$workId`; this endpoint
ships everything that page needs in one round-trip:

  - Work identity (`cross_refs`) — drives the "Manga này ở các nguồn
    nào" section, plus link-vote UI in a later commit.
  - Sibling materials — per-source metadata (title, cover, status,
    language). The SPA picks one "active" material via `?src=` URL
    state and fetches its manifest chapter list live; the others
    are listed as switch targets without manifest fetches.
  - Cross-source chapter overlay — every shared translation in the
    Work, grouped by `work_chapter` (= number_norm). Drives the
    "[VN] @userA · MangaDex" badge on each chapter row regardless
    of which sibling spawned the draft.
  - Viewer's library entry, when present.

Per "manifest-first" architecture (see plan): server never lists the
chapter set of a Work — that's the manifest's authority. We only
list `work_chapters` the community has touched (spawn / upload / raw
history) so the SPA can overlay translations against the manifest's
live chapter list by matching `number_norm`.

Community link voting:

    GET  /api/work/{id}/link-suggestions
    POST /api/work/{id}/link-vote        body: {target_material_id, vote: ±1}
    POST /api/work/{id}/propose-link     body: {target_material_id}
    POST /api/work/{id}/force-link       body: {target_material_id}

Votes cross the `LINK_MERGE_THRESHOLD` (default 2 distinct users) →
inline merge of the two Works, refused when both Works carry
conflicting `cross_refs`. The threshold is intentionally a constant,
not user-tunable.

`force-link` bypasses the threshold for the explicit manual-link
flow: when one user picks a candidate via search and confirms the
modal, the merge runs immediately. The +1 vote is still recorded
for the audit trail and the cross_refs conflict check still applies.
The community vote path keeps threshold semantics.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from typoon.api.deps import (
    get_auth_cfg, get_config, get_inbox, get_store, require_user,
)
from typoon.api.models import (
    LinkSuggestionOut, LinkVoteResult, SplitVoteResult,
    MaterialOut, UploadingChapter, WorkChapterOut, WorkChapterTranslation,
    WorkDetailOut, WorkMemberOut, WorkOut, WorkViewerEntry,
)
from typoon.api.routes._shared import require_material, require_work
from typoon.api.routes.upload import (
    UploadInitBody, UploadInitOut, _init_upload_for_material,
)
from typoon.adapters.inbox import ChapterInbox
from typoon.config import AuthConfig, Config
from typoon.storage.store import Store


router = APIRouter(
    prefix="/api/work", tags=["work"],
    dependencies=[Depends(require_user)],
)


# Distinct-user vote threshold for inline Work merging. Kept here as
# a constant; tuning this is a moderation-level decision, not
# per-user.
#
# Beta scale: a 2-user vote is enough to merge. The deployment is a
# Discord-guild closed community (~tens of users); requiring three
# distinct votes meant cross-language links never materialized in
# practice because users would link once and forget. Two votes still
# guards against a single bad actor merging unrelated manga but
# closes the "I linked it and nothing happened" gap.
LINK_MERGE_THRESHOLD = 2


# Mirror — same gating shape, inverse operation. Two distinct +1
# votes (or one explicit owner force_unlink within the undo window)
# move a material out of its current Work into a fresh isolated one.
SPLIT_THRESHOLD = 2

# Owner-only undo window for force_link / force_unlink. After this
# many minutes the only path back is the regular split-vote flow.
FORCE_UNDO_WINDOW_MIN = 10


@router.get("/{work_id}", response_model=WorkDetailOut)
async def get_work_detail(
    work_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    work = await require_work(work_id, db)
    materials = await db.list_materials_for_work(work_id)
    chapters = await db.list_work_chapters_with_translations(
        work_id, viewer_id=user["id"],
    )
    entry = await db.find_entry_for_work(
        user_id=user["id"], work_id=work_id,
    )
    return WorkDetailOut(
        work=WorkOut(
            id=int(work["id"]),
            cross_refs=work.get("cross_refs"),
            created_at=work.get("created_at"),
            updated_at=work.get("updated_at"),
        ),
        materials=[MaterialOut.from_row(m) for m in materials],
        chapters=[
            WorkChapterOut(
                id=int(c["id"]),
                number_norm=c["number_norm"],
                label=c.get("label"),
                translations=[
                    WorkChapterTranslation(**t) for t in c["translations"]
                ],
                uploading_chapters=[
                    UploadingChapter(**u) for u in c["uploading_chapters"]
                ],
            )
            for c in chapters
        ],
        viewer_entry=(
            WorkViewerEntry(
                entry_id=int(entry["id"]),
                status=entry["status"],
                target_lang=entry["target_lang"],
            )
            if entry is not None else None
        ),
    )


class CreateBlankWorkBody(BaseModel):
    """Body for `POST /api/work` — viewer-driven "Tạo trống".

    Creates an empty Work plus a library entry for the viewer. Upload
    materials are not seeded here; the first chapter upload will
    lazily create one. `target_lang` defaults to the user's
    `preferred_target_lang` when omitted (resolved at the route
    level)."""
    title:       str         = Field(min_length=1, max_length=200)
    cover_url:   str | None  = None
    target_lang: str | None  = None


@router.post("", response_model=WorkDetailOut, status_code=201)
async def create_blank_work(
    body: CreateBlankWorkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Create an empty Work that the viewer can follow before any
    source plugin indexes it. Used for the "+ Tạo trống" flow on
    /library — user knows the manga, no source has it yet, viewer
    wants to bookmark or upload chapters manually.

    Returns the full `WorkDetailOut` so the SPA can navigate straight
    to `/w/$new_id` without a follow-up GET.
    """
    target_lang = (body.target_lang
                   or user.get("preferred_target_lang")
                   or 'vi')
    work_id, _ = await db.create_blank_work(
        user_id=user["id"],
        title=body.title,
        cover_url=body.cover_url,
        target_lang=target_lang,
    )
    # Re-fetch via the same helper the GET uses so the response shape
    # stays in lockstep with /work/{id}.
    work = await require_work(work_id, db)
    entry = await db.find_entry_for_work(
        user_id=user["id"], work_id=work_id,
    )
    return WorkDetailOut(
        work=WorkOut(
            id=int(work["id"]),
            cross_refs=work.get("cross_refs"),
            created_at=work.get("created_at"),
            updated_at=work.get("updated_at"),
        ),
        materials=[],
        chapters=[],
        viewer_entry=(
            WorkViewerEntry(
                entry_id=int(entry["id"]),
                status=entry["status"],
                target_lang=entry["target_lang"],
            )
            if entry is not None else None
        ),
    )


@router.delete("/{work_id}", status_code=204)
async def delete_user_work(
    work_id: int,
    user:  dict  = Depends(require_user),
    db:    Store = Depends(get_store),
):
    """Delete a user-created Work (origin='upload' materials only).

    Only allowed when the Work has no source-backed materials —
    those are shared community data and must not be deleted by a
    single user. Upload/extension materials are user-owned and can
    be removed with the Work.

    Cascade: deletes all upload/extension materials, their chapters,
    and all library entries for this Work belonging to this user.
    Raises 403 if the Work has any source material.
    Raises 404 if the Work does not exist.
    """
    await db.delete_user_work(work_id=work_id, user_id=user["id"])


@router.delete("/{work_id}/my-upload", status_code=204)
async def delete_my_upload_material(
    work_id: int,
    user:  dict  = Depends(require_user),
    db:    Store = Depends(get_store),
):
    """Delete the viewer's upload-origin material for a Work.

    Used when unfollowing a shared work to clean up the viewer's own
    uploaded chapters. The work and other materials are left intact.
    No-op (204) if the viewer has no upload material on this work.
    """
    await db.delete_user_upload_material(work_id=work_id, user_id=user["id"])


@router.post(
    "/{work_id}/upload-init",
    response_model=UploadInitOut,
)
async def work_upload_init(
    work_id: int,
    body:    UploadInitBody,
    user:  dict          = Depends(require_user),
    db:    Store         = Depends(get_store),
    cfg:   Config        = Depends(get_config),
    auth:  AuthConfig    = Depends(get_auth_cfg),
    inbox: ChapterInbox  = Depends(get_inbox),
):
    """Per-work upload entry point — lazy-create the viewer's
    upload-origin material on first call, then delegate to the
    shared upload-init helper.

    The SPA's "Tải lên chương" button targets `/api/work/{id}/upload-
    init` regardless of whether the user has uploaded before. Server
    resolves (or creates) the unique `(imported_by, work_id) WHERE
    origin='upload'` material — same row reused on every subsequent
    chapter upload — and surfaces its id in the response so the SDK
    can finalize against `/api/material/{material_id}/chapter/...`.

    Title + cover for the upload material inherit from the Work's
    canonical display snapshot (any sibling source material's title,
    fallback to the viewer's library-entry title). Single-row per
    user × work means user can't accidentally fan out spam materials.
    """
    await require_work(work_id, db)
    material_id = await db.get_or_create_upload_material(
        work_id=work_id, imported_by=user["id"],
    )
    return await _init_upload_for_material(
        material_id=material_id,
        byte_size=body.byte_size,
        user=user, db=db, cfg=cfg, auth=auth, inbox=inbox,
    )


@router.get(
    "/{work_id}/link-suggestions",
    response_model=list[LinkSuggestionOut],
)
async def list_link_suggestions(
    work_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Cross-source link candidates the SPA can offer for this Work.

    Two streams merged into one list:

      1. Already-voted pairs (`kind='voted'`) — community has cast at
         least one +1 vote; we surface the aggregate score so the
         viewer can pile on or veto.
      2. Title-similarity ranker (`kind='ranked'`) — pg_trgm + a few
         signal bonuses surface candidates BEFORE any vote exists, so
         the panel isn't empty on a fresh manga. Filtered to drop
         anything already on stream 1 to avoid duplicates.

    The viewer's own vote (when present) surfaces in each row so the
    UI can render "Đã đồng ý" / "Đã từ chối" instead of the +1/−1
    buttons.
    """
    await require_work(work_id, db)

    voted_rows   = await db.list_work_link_suggestions(work_id=work_id)
    voted_keys   = {int(r["candidate_material_id"]) for r in voted_rows}
    ranked_rows  = await db.list_work_link_candidates(work_id=work_id)

    out: list[LinkSuggestionOut] = []

    # Stream 1: existing votes — the more authoritative signal.
    for r in voted_rows:
        own_id  = int(r["own_material_id"])
        cand_id = int(r["candidate_material_id"])
        viewer_vote = await db.get_link_vote(
            voter_id=user["id"],
            material_a_id=own_id,
            material_b_id=cand_id,
        )
        out.append(LinkSuggestionOut(
            kind="voted",
            candidate_material_id=cand_id,
            candidate_title=r["candidate_title"],
            candidate_source=r.get("candidate_source"),
            candidate_cover=r.get("candidate_cover"),
            candidate_work_id=int(r["candidate_work_id"]),
            own_material_id=own_id,
            score=int(r["score"]),
            total_votes=int(r["total"]),
            viewer_vote=viewer_vote,
        ))

    # Stream 2: ranker candidates — skip anything already voted.
    for r in ranked_rows:
        cand_id = int(r["candidate_material_id"])
        if cand_id in voted_keys:
            continue
        own_id  = int(r["own_material_id"])
        viewer_vote = await db.get_link_vote(
            voter_id=user["id"],
            material_a_id=own_id,
            material_b_id=cand_id,
        )
        out.append(LinkSuggestionOut(
            kind="ranked",
            candidate_material_id=cand_id,
            candidate_title=r["candidate_title"],
            candidate_source=r.get("candidate_source"),
            candidate_cover=r.get("candidate_cover"),
            candidate_work_id=int(r["candidate_work_id"]),
            own_material_id=own_id,
            confidence=float(r["score"]),
            reason=r["reason"],
            viewer_vote=viewer_vote,
        ))

    return out


class LinkVoteBody(BaseModel):
    """Cast a +1 / −1 vote on a (this work × candidate material) pair.

    The vote is implicitly between `target_material_id` and the
    sibling material the SPA surfaced as the trigger — server picks
    that sibling via `own_material_id`, defaulting to the first
    material in the Work when omitted.
    """
    target_material_id: int
    vote:               Literal[-1, 1]
    own_material_id:    int | None = None


async def _resolve_link_pair(
    *,
    work_id:            int,
    target_material_id: int,
    own_material_id:    int | None,
    db:                 Store,
) -> tuple[int, int]:
    """Validate the (work, target, own) trio shared by every link
    endpoint and return ``(own_id, target_id)``. Raises HTTPException
    on the same conditions as the original inline checks."""
    await require_work(work_id, db)
    target = await require_material(target_material_id, db)

    # Pick the sibling material that anchors this vote. Default: the
    # first material attached to the work (oldest). Caller may pin a
    # specific one (e.g. the user is browsing "the HappyMH version").
    if own_material_id is not None:
        own = await require_material(own_material_id, db)
        if int(own["work_id"]) != work_id:
            raise HTTPException(
                400, "own_material_id does not belong to this work",
            )
        own_id = int(own["id"])
    else:
        mats = await db.list_materials_for_work(work_id)
        if not mats:
            raise HTTPException(409, "Work has no materials")
        own_id = int(mats[0]["id"])

    if int(target["id"]) == own_id:
        raise HTTPException(400, "Cannot link a material to itself")
    if int(target["work_id"]) == work_id:
        raise HTTPException(
            409, "target material already belongs to this work",
        )
    return own_id, int(target["id"])


def _link_vote_result(result: dict) -> LinkVoteResult:
    return LinkVoteResult(
        vote=int(result["vote"]),
        score=int(result["score"]),
        merged=bool(result["merged"]),
        canonical_work_id=result.get("canonical_work_id"),
        blocked_reason=result.get("blocked_reason"),
    )


@router.post(
    "/{work_id}/link-vote",
    response_model=LinkVoteResult,
)
async def cast_link_vote(
    work_id: int,
    body:    LinkVoteBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Cast / update the viewer's link vote. Merge may fire inline
    when score crosses LINK_MERGE_THRESHOLD."""
    own_id, target_id = await _resolve_link_pair(
        work_id=work_id,
        target_material_id=body.target_material_id,
        own_material_id=body.own_material_id,
        db=db,
    )
    result = await db.cast_link_vote_with_merge(
        voter_id=user["id"],
        material_a_id=own_id,
        material_b_id=target_id,
        vote=body.vote,
        threshold=LINK_MERGE_THRESHOLD,
    )
    return _link_vote_result(result)


class ProposeLinkBody(BaseModel):
    """Surface a new candidate the community hasn't voted on yet.
    Equivalent to `cast_link_vote(vote=+1)` for the proposing user.
    """
    target_material_id: int
    own_material_id:    int | None = None


@router.post(
    "/{work_id}/propose-link",
    response_model=LinkVoteResult,
)
async def propose_link(
    work_id: int,
    body:    ProposeLinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Convenience wrapper — cast `+1` on a candidate the SPA's
    search just produced. Same result shape as `link-vote`."""
    return await cast_link_vote(
        work_id,
        LinkVoteBody(
            target_material_id=body.target_material_id,
            vote=1,
            own_material_id=body.own_material_id,
        ),
        user=user, db=db,
    )


class ForceLinkBody(BaseModel):
    """Body for the explicit manual-link path.

    Same shape as `ProposeLinkBody`; lives as a distinct type so the
    intent is self-documenting at call sites and easy to extend
    (audit reason, etc.) without churning the vote schema.
    """
    target_material_id: int
    own_material_id:    int | None = None


@router.post(
    "/{work_id}/force-link",
    response_model=LinkVoteResult,
)
async def force_link(
    work_id: int,
    body:    ForceLinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Explicit manual link: the viewer affirmatively chose this
    pair via search and wants the two Works merged immediately.
    Bypasses the community-vote threshold but still records a +1
    vote (so the audit trail mirrors the regular path) and still
    refuses merges across conflicting `cross_refs`."""
    own_id, target_id = await _resolve_link_pair(
        work_id=work_id,
        target_material_id=body.target_material_id,
        own_material_id=body.own_material_id,
        db=db,
    )
    result = await db.cast_link_vote_with_merge(
        voter_id=user["id"],
        material_a_id=own_id,
        material_b_id=target_id,
        vote=1,
        threshold=LINK_MERGE_THRESHOLD,
        force_merge=True,
    )
    # Audit the action so the actor can undo within the window. We
    # only log on a successful merge; a refused force_link (e.g.
    # cross_refs_conflict) has nothing to undo.
    if result.get("merged") and result.get("canonical_work_id") is not None:
        await db.log_force_action(
            actor_id=user["id"],
            kind="force_link",
            material_a_id=own_id,
            material_b_id=target_id,
            target_work_id=int(result["canonical_work_id"]),
        )
    return _link_vote_result(result)


# ── Split / unlink ─────────────────────────────────────────────────


class SplitVoteBody(BaseModel):
    """Body for `POST /api/work/{id}/split-vote`. `material_id` must
    currently belong to this Work; otherwise the route returns 400."""
    material_id: int
    vote:        Literal[-1, 1]


class ForceUnlinkBody(BaseModel):
    """Body for `POST /api/work/{id}/force-unlink`. The viewer must
    be the actor of a non-reversed `force_link` involving this
    material that fired within `FORCE_UNDO_WINDOW_MIN` minutes."""
    material_id: int


def _split_vote_result(result: dict) -> SplitVoteResult:
    return SplitVoteResult(
        vote=int(result["vote"]),
        score=int(result["score"]),
        split=bool(result["split"]),
        new_work_id=result.get("new_work_id"),
        blocked_reason=result.get("blocked_reason"),
    )


async def _require_member(
    work_id: int, material_id: int, db: Store,
) -> dict:
    """Resolve `material_id` and assert it currently belongs to
    `work_id`. Same shape as `_resolve_link_pair` but for the inverse
    operation: split / unlink act on a single member of the Work."""
    await require_work(work_id, db)
    material = await require_material(material_id, db)
    if int(material["work_id"]) != work_id:
        raise HTTPException(
            400, "material does not belong to this work",
        )
    return material


@router.post(
    "/{work_id}/split-vote",
    response_model=SplitVoteResult,
)
async def cast_split_vote(
    work_id: int,
    body:    SplitVoteBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Community vote: "this material is in the wrong work, tách
    nó ra". Mirrors `link-vote` but inverted — the score gates an
    inline move of `material_id` into a fresh isolated Work."""
    await _require_member(work_id, body.material_id, db)
    result = await db.cast_split_vote_with_split(
        voter_id=user["id"],
        material_id=body.material_id,
        vote=body.vote,
        threshold=SPLIT_THRESHOLD,
    )
    return _split_vote_result(result)


@router.post(
    "/{work_id}/force-unlink",
    response_model=SplitVoteResult,
)
async def force_unlink(
    work_id: int,
    body:    ForceUnlinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Owner undo for a recent `force_link`. Only the actor of a
    non-reversed force_link within `FORCE_UNDO_WINDOW_MIN` may call
    this; after the window the route returns 403 and the material
    can only leave via the regular split-vote flow.

    Successful unlink moves the material to a fresh isolated Work
    and returns the same `SplitVoteResult` shape so the SPA's
    success handler stays unified across the two flows.
    """
    await _require_member(work_id, body.material_id, db)
    recent = await db.get_recent_force_link(
        actor_id=user["id"],
        material_id=body.material_id,
        window_minutes=FORCE_UNDO_WINDOW_MIN,
    )
    if recent is None:
        raise HTTPException(
            403,
            "Không có liên kết gần đây để hoàn tác. "
            "Dùng 'Báo nhầm nguồn' để đề xuất tách.",
        )
    try:
        res = await db.force_unlink_material(
            actor_id=user["id"], material_id=body.material_id,
        )
    except ValueError as e:
        if str(e) == "solo_member":
            return SplitVoteResult(
                vote=1, score=0, split=False,
                new_work_id=None,
                blocked_reason="solo_member",
            )
        raise
    return SplitVoteResult(
        vote=1, score=0, split=True,
        new_work_id=int(res["new_work_id"]),
        blocked_reason=None,
    )


@router.get(
    "/{work_id}/members",
    response_model=list[WorkMemberOut],
)
async def list_members(
    work_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Materials currently attached to this Work + the viewer's
    split-vote state on each + the owner-undo hint when applicable.

    Drives the "Nguồn đang đọc" panel on the Work hub. Single
    round-trip — the SPA renders the whole panel from this payload
    without per-row fetches.
    """
    await require_work(work_id, db)
    materials = await db.list_materials_for_work(work_id)

    out: list[WorkMemberOut] = []
    for m in materials:
        mid = int(m["id"])
        viewer_vote = await db.get_split_vote(
            voter_id=user["id"], material_id=mid,
        )
        score = await db.get_split_score(mid)
        recent = await db.get_recent_force_link(
            actor_id=user["id"],
            material_id=mid,
            window_minutes=FORCE_UNDO_WINDOW_MIN,
        )
        undo_expires: str | None = None
        if recent is not None:
            # Format the absolute expiry the SPA can render a
            # countdown against without server time drift creeping
            # in per render.
            from datetime import timedelta
            expires = (
                recent["created_at"]
                + timedelta(minutes=FORCE_UNDO_WINDOW_MIN)
            )
            undo_expires = expires.strftime("%Y-%m-%dT%H:%M:%SZ")

        out.append(WorkMemberOut(
            material_id              = mid,
            title                    = m["title"],
            cover_url                = m.get("cover_url"),
            source                   = m.get("source"),
            languages                = list(m.get("languages") or []),
            title_native             = m.get("title_native"),
            title_locale             = m.get("title_locale") or None,
            viewer_split_vote        = viewer_vote,
            pending_split_score      = int(score["score"]),
            pending_split_threshold  = SPLIT_THRESHOLD,
            force_link_undo_expires_at = undo_expires,
        ))
    return out

