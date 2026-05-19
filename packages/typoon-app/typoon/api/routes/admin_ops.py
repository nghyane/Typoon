"""Admin / ops dashboard — pipeline queue + audit.

Surface for the Discord-elevated admin to recover the translation
pipeline without SSH: pause/resume stages, requeue dead-letter tasks,
release stale claims, force-fail unrecoverable rows, and read the
audit trail of every mutation. Every endpoint is `Depends(require_admin)`.

Concurrency model — important enough to call out:

  Task mutations carry the operator's view of (attempts, claimed_by)
  in the body. The store checks those against the live row inside a
  `SELECT ... FOR UPDATE` transaction; a mismatch yields 409, which
  the SPA turns into a "trạng thái đã thay đổi, refresh" toast. The
  pattern keeps two simultaneous admins from clobbering each other,
  and keeps a worker that picked the task up between snapshot and
  click from losing its claim.

Idempotency:

  Every mutation accepts an `Idempotency-Key` header. When present,
  it is recorded under `admin_actions.target_ref.idem_key` with a
  unique partial index — a network retry of the same click produces
  one audit row and one state change, never two. Absent header = no
  dedup, every call is treated as new.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from typoon.api.deps import get_store, require_admin
from typoon.api.models import (
    AdminActionLit,
    AdminActionOut,
    ForceFailTaskIn,
    PausedStageOut,
    PipelineStageLit,
    ReleaseTaskIn,
    RequeueTaskIn,
    StagePauseIn,
    StageResumeIn,
    TaskListOut,
    TaskOut,
    TaskStateLit,
    TaskTargetKindLit,
)
from typoon.storage.store import Store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/ops", tags=["admin"],
    dependencies=[Depends(require_admin)],
)


# ── Stages ───────────────────────────────────────────────────────────


@router.get("/stages", response_model=list[PausedStageOut])
async def list_paused_stages(db: Store = Depends(get_store)):
    """Snapshot of `stage_pause`. Stages absent from the response are
    healthy — workers are claiming from them normally."""
    rows = await db.list_paused_stages()
    return [
        PausedStageOut(
            stage     = r["stage"],
            reason    = r["reason"],
            paused_at = r["paused_at"].isoformat(),
            paused_by = r["paused_by"],
        )
        for r in rows
    ]


@router.post("/stages/{stage}/pause", status_code=204)
async def pause_stage(
    stage:           PipelineStageLit,
    body:            StagePauseIn,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    user:            dict       = Depends(require_admin),
    db:              Store      = Depends(get_store),
):
    """Manually pause a stage (workers stop claiming new tasks).

    Idempotent: pausing an already-paused stage returns 204 (no audit
    row written, the existing pause keeps its original reason). The
    operator who wants to *change* the pause reason must resume + pause
    again so the audit trail tells the full story."""
    await db.pause_stage(
        stage      = stage,
        reason     = body.reason,
        actor_id   = user["id"],
        source     = f"web:user:{user['id']}",
        idem_key   = idempotency_key,
    )
    return None


@router.post("/stages/{stage}/resume", status_code=204)
async def resume_stage(
    stage:           PipelineStageLit,
    body:            StageResumeIn,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    user:            dict       = Depends(require_admin),
    db:              Store      = Depends(get_store),
):
    """Lift a pause. Workers wake on the resume NOTIFY and start
    claiming again. Resuming an unpaused stage returns 204 silently
    (idempotent)."""
    await db.resume_stage(
        stage      = stage,
        reason     = body.reason,
        actor_id   = user["id"],
        source     = f"web:user:{user['id']}",
        idem_key   = idempotency_key,
    )
    return None


# ── Tasks ────────────────────────────────────────────────────────────


@router.get("/tasks", response_model=TaskListOut)
async def list_tasks(
    stage:       PipelineStageLit  | None = Query(None),
    state:       TaskStateLit      | None = Query(None),
    target_kind: TaskTargetKindLit | None = Query(None),
    limit:       int                      = Query(50, ge=1, le=200),
    cursor:      str               | None = Query(None),
    db:          Store                    = Depends(get_store),
):
    """Paged queue snapshot with derived `lifecycle_state` per row.
    Keyset cursor stays stable while workers churn claims."""
    page = await db.list_queue_tasks(
        stage=stage, state=state, target_kind=target_kind,
        limit=limit, cursor=cursor,
    )
    return TaskListOut(
        items=[_task_out(r) for r in page["items"]],
        next_cursor=page["next_cursor"],
    )


@router.post(
    "/tasks/{stage}/{target_kind}/{target_id}/requeue",
    status_code=204,
)
async def requeue_task(
    stage:            PipelineStageLit,
    target_kind:      TaskTargetKindLit,
    target_id:        int,
    body:             RequeueTaskIn,
    idempotency_key:  str | None = Header(None, alias="Idempotency-Key"),
    user:             dict       = Depends(require_admin),
    db:               Store      = Depends(get_store),
):
    """Reset a dead-letter row so workers will retry it.

    Optimistic concurrency: `expected_attempts` + `expected_claimed_by`
    in the body must match the live row, or the call returns 409 and
    the admin re-fetches the dashboard. 404 if the task is gone."""
    ok = await db.requeue_task(
        target_kind         = target_kind,
        target_id           = target_id,
        stage               = stage,
        expected_attempts   = body.expected_attempts,
        expected_claimed_by = body.expected_claimed_by,
        reason              = body.reason,
        actor_id            = user["id"],
        source              = f"web:user:{user['id']}",
        idem_key            = idempotency_key,
    )
    _ensure_mutation_landed(ok)
    return None


@router.post(
    "/tasks/{stage}/{target_kind}/{target_id}/release",
    status_code=204,
)
async def release_task(
    stage:            PipelineStageLit,
    target_kind:      TaskTargetKindLit,
    target_id:        int,
    body:             ReleaseTaskIn,
    idempotency_key:  str | None = Header(None, alias="Idempotency-Key"),
    user:             dict       = Depends(require_admin),
    db:               Store      = Depends(get_store),
):
    """Clear a stale claim (worker died without finishing).
    Attempts untouched — the task picks up where it was."""
    ok = await db.release_task_claim(
        target_kind         = target_kind,
        target_id           = target_id,
        stage               = stage,
        expected_claimed_by = body.expected_claimed_by,
        reason              = body.reason,
        actor_id            = user["id"],
        source              = f"web:user:{user['id']}",
        idem_key            = idempotency_key,
    )
    _ensure_mutation_landed(ok)
    return None


@router.post(
    "/tasks/{stage}/{target_kind}/{target_id}/fail",
    status_code=204,
)
async def force_fail_task(
    stage:            PipelineStageLit,
    target_kind:      TaskTargetKindLit,
    target_id:        int,
    body:             ForceFailTaskIn,
    idempotency_key:  str | None = Header(None, alias="Idempotency-Key"),
    user:             dict       = Depends(require_admin),
    db:               Store      = Depends(get_store),
):
    """Mark a task dead-lettered on operator decision (corrupt source,
    duplicate work, etc). The row stays for forensic — admin can
    requeue later if context changes."""
    ok = await db.force_fail_task(
        target_kind         = target_kind,
        target_id           = target_id,
        stage               = stage,
        expected_attempts   = body.expected_attempts,
        expected_claimed_by = body.expected_claimed_by,
        reason              = body.reason,
        actor_id            = user["id"],
        source              = f"web:user:{user['id']}",
        idem_key            = idempotency_key,
    )
    _ensure_mutation_landed(ok)
    return None


# ── Drafts ───────────────────────────────────────────────────────────


class AdminRestartDraftIn(BaseModel):
    """Body for `POST /api/admin/ops/drafts/{draft_id}/restart`.

    `reason` is required so every restart has a written justification
    in the audit log — admins should not be able to silently re-kick
    pipelines."""
    reason: str


@router.post(
    "/drafts/{draft_id}/restart",
    status_code=204,
)
async def restart_draft(
    draft_id:        int,
    body:            AdminRestartDraftIn,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    user:            dict       = Depends(require_admin),
    db:              Store      = Depends(get_store),
):
    """Force-restart a translation draft from its entry stage,
    regardless of current state.

    This is the admin equivalent of the user-facing `POST
    /api/translate/{id}/redo`, with three differences:

      - Works on any state, including `done` and `blocked`. User
        redo refuses those (done is canonical, blocked is a policy
        hold). Admin can override both — e.g. a model upgrade
        invalidates a `done` output, or a blocked draft was
        cleared by moderation review.
      - No owner check. Admin acts on any user's draft.
      - Writes an `admin_actions.draft.restart` row with the
        previous draft state captured for forensic.

    All translations pinned to this draft will see the new render
    once the pipeline finishes — `archive_*` columns are overwritten
    in place, so existing `translation_id` deep links stay valid.
    No takedown / row-swap dance is needed for that to work.

    Idempotent on `Idempotency-Key` header (collapses to one audit
    row + one restart). Returns 404 if the draft has been deleted,
    204 on success.
    """
    draft = await db.get_draft(draft_id)
    if draft is None:
        raise HTTPException(404, "Draft not found")
    chapter = await db.get_chapter(int(draft["chapter_id"]))
    if chapter is None:
        raise HTTPException(500, "Draft references missing chapter")

    # Same entry-stage logic as user spawn/redo, imported from the
    # translate route so the two paths cannot drift on what "entry
    # stage" means for a chapter at a given prepared/scan state.
    from typoon.api.routes.translate import _resolve_entry_task
    stage, kind, tid = await _resolve_entry_task(db, chapter, draft_id)

    ok = await db.restart_draft_pipeline(
        draft_id,
        entry_stage=stage,
        entry_target_kind=kind,
        entry_target_id=tid,
        audit_actor_id=user["id"],
        audit_reason=body.reason,
        audit_source=f"web:user:{user['id']}",
        audit_idem_key=idempotency_key,
    )
    _ensure_mutation_landed(ok)
    return None


# ── Audit ────────────────────────────────────────────────────────────


@router.get("/actions", response_model=list[AdminActionOut])
async def list_actions(
    action:      AdminActionLit    | None = Query(None),
    actor_id:    int               | None = Query(None),
    stage:       PipelineStageLit  | None = Query(None),
    target_kind: TaskTargetKindLit | None = Query(None),
    target_id:   int               | None = Query(None),
    limit:       int                      = Query(50, ge=1, le=200),
    before_id:   int               | None = Query(None),
    db:          Store                    = Depends(get_store),
):
    """Reverse-chronological audit feed. `before_id` is the keyset
    cursor (id DESC); pass the last seen `id` to fetch older entries."""
    rows = await db.list_admin_actions(
        action=action, actor_id=actor_id,
        stage=stage, target_kind=target_kind, target_id=target_id,
        limit=limit, before_id=before_id,
    )
    return [
        AdminActionOut(
            id         = r["id"],
            at         = r["at"].isoformat(),
            actor_id   = r["actor_id"],
            action     = r["action"],
            target_ref = r["target_ref"],
            reason     = r["reason"],
            prev_state = r["prev_state"],
        )
        for r in rows
    ]


# ── Helpers ──────────────────────────────────────────────────────────


def _task_out(r: dict) -> TaskOut:
    return TaskOut(
        stage             = r["stage"],
        target_kind       = r["target_kind"],
        target_id         = r["target_id"],
        attempts          = r["attempts"],
        claimed_by        = r["claimed_by"],
        claimed_at        = r["claimed_at"].isoformat() if r["claimed_at"] else None,
        last_error        = r["last_error"],
        lifecycle_state   = r["lifecycle_state"],
        claim_age_seconds = r["claim_age_seconds"],
        work_id           = r.get("work_id"),
        work_chapter_id   = r.get("work_chapter_id"),
        chapter_id        = r.get("chapter_id"),
        chapter_label     = r.get("chapter_label"),
        source_lang       = r.get("source_lang"),
        target_lang       = r.get("target_lang"),
        llm_model         = r.get("llm_model"),
        owner_id          = r.get("owner_id"),
    )


def _ensure_mutation_landed(ok: bool) -> None:
    """Translate the store's boolean into the HTTP error spectrum.

    The store returns False for two cases the operator should react
    to identically: the row vanished (404-ish) or the optimistic
    guard rejected (409-ish). Both mean "your snapshot was stale,
    refresh and decide again", so we collapse to 409 — the dashboard
    will re-fetch either way."""
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="State has changed since the snapshot — refresh and retry",
        )
