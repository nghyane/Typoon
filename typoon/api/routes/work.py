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

Votes cross the `LINK_MERGE_THRESHOLD` (default 3 distinct users) →
inline merge of the two Works, refused when both Works carry
conflicting `cross_refs`. The threshold is intentionally a constant,
not user-tunable.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.models import (
    LinkSuggestionOut, LinkVoteResult,
    MaterialOut, WorkChapterOut, WorkChapterTranslation, WorkDetailOut,
    WorkOut, WorkViewerEntry,
)
from typoon.api.routes._shared import require_material, require_work
from typoon.storage import Store


router = APIRouter(
    prefix="/api/work", tags=["work"],
    dependencies=[Depends(require_user)],
)


# Distinct-user vote threshold for inline Work merging. Kept here as a
# constant; tuning this is a moderation-level decision, not per-user.
LINK_MERGE_THRESHOLD = 3


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


# ── Community link voting ────────────────────────────────────────────


@router.get(
    "/{work_id}/link-suggestions",
    response_model=list[LinkSuggestionOut],
)
async def list_link_suggestions(
    work_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Materials outside this Work that have a positive community
    link-vote score with any sibling inside it.

    The viewer's own vote (when present) surfaces in the row so the
    UI can render "Đã đồng ý" / "Đã từ chối" instead of the +1/−1
    buttons.
    """
    await require_work(work_id, db)
    rows = await db.list_work_link_suggestions(work_id=work_id)
    out: list[LinkSuggestionOut] = []
    for r in rows:
        own_id = int(r["own_material_id"])
        cand_id = int(r["candidate_material_id"])
        viewer_vote = await db.get_link_vote(
            voter_id=user["id"],
            material_a_id=own_id,
            material_b_id=cand_id,
        )
        out.append(LinkSuggestionOut(
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
    await require_work(work_id, db)
    target = await require_material(body.target_material_id, db)

    # Pick the sibling material that anchors this vote. Default: the
    # first material attached to the work (oldest). Caller may pin a
    # specific one (e.g. the user is browsing "the HappyMH version").
    if body.own_material_id is not None:
        own = await require_material(body.own_material_id, db)
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

    result = await db.cast_link_vote_with_merge(
        voter_id=user["id"],
        material_a_id=own_id,
        material_b_id=int(target["id"]),
        vote=body.vote,
        threshold=LINK_MERGE_THRESHOLD,
    )
    return LinkVoteResult(
        vote=int(result["vote"]),
        score=int(result["score"]),
        merged=bool(result["merged"]),
        canonical_work_id=result.get("canonical_work_id"),
        blocked_reason=result.get("blocked_reason"),
    )


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

