"""Reports + moderation routes — public intake + admin action log.

Two surfaces, sharply separated:

  - `/api/reports`          authenticated users submit. Open queue
                            visible only to admin via the admin
                            sub-router.
  - `/api/admin/reports`    admin reads the open queue and acts on
                            each report via `/actions`. Actions
                            also accept direct (report-less)
                            takedowns for proactive cleanup.

Read-path visibility is still gated by `takedown_at` on the target
row. `moderation_actions` is the audit trail of how those flags got
flipped.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from typoon.api.deps import get_store, require_admin, require_user
from typoon.api.models import (
    ModerationActionOut, ReportOut,
    ReportTargetKind, ReportKind, ReportStatus, ModerationAction,
)
from typoon.storage.store import Store


router = APIRouter(prefix="/api/reports", tags=["reports"])


class SubmitReportBody(BaseModel):
    target_kind:    ReportTargetKind
    target_id:      int
    # Free-form category. `dmca` is the legal route; `abuse` covers
    # harassment / NSFW-in-SFW; `quality` lets users flag bad
    # translations without it being a takedown signal.
    kind:           ReportKind  = "dmca"
    reason:         str         = Field(min_length=4, max_length=4000)


@router.post("", status_code=202)
async def submit(
    body: SubmitReportBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Anyone authenticated can file a report. Does NOT auto-takedown
    — admin moderation queue picks it up."""
    report_id = await db.submit_report(
        reporter_id=user["id"],
        reporter_label=user.get("display_name") or f"user:{user['id']}",
        target_kind=body.target_kind,
        target_id=body.target_id,
        kind=body.kind,
        reason=body.reason,
    )
    return {"report_id": report_id}


admin_router = APIRouter(
    prefix="/api/admin/reports", tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@admin_router.get("", response_model=list[ReportOut])
async def list_admin_reports(
    status: ReportStatus | None = Query(None),
    limit:  int                 = Query(100, ge=1, le=500),
    db: Store = Depends(get_store),
):
    rows = await db.list_reports(status=status, limit=limit)
    return [ReportOut(**r) for r in rows]


@admin_router.get("/{report_id}", response_model=ReportOut)
async def get_admin_report(
    report_id: int,
    db: Store = Depends(get_store),
):
    row = await db.get_report(report_id)
    if row is None:
        raise HTTPException(404, "Report not found")
    return ReportOut(**row)


class StatusBody(BaseModel):
    status: ReportStatus


@admin_router.patch("/{report_id}/status", status_code=204)
async def patch_admin_report_status(
    report_id: int,
    body:      StatusBody,
    user:      dict  = Depends(require_admin),
    db:        Store = Depends(get_store),
):
    ok = await db.update_report_status(
        report_id, status=body.status, resolver_id=user["id"],
    )
    if not ok:
        raise HTTPException(404, "Report not found")


class ActionBody(BaseModel):
    # `report_id` optional: admin can act proactively (no triggering
    # report) — common for crawler-detected scraping bulk-takedowns.
    report_id:   int | None = None
    target_kind: ReportTargetKind
    target_id:   int
    action:      ModerationAction
    reason:      str = Field(min_length=4, max_length=4000)


@admin_router.post("/actions", response_model=ModerationActionOut)
async def apply_action(
    body: ActionBody,
    user: dict  = Depends(require_admin),
    db:   Store = Depends(get_store),
):
    """Execute and log a moderation action.

    Semantics:
      - takedown on draft / translation  → set takedown_at + reason.
      - restore  on draft / translation  → clear takedown_at + reason.
      - delete   on material / chapter   → hard delete (cascade).

    The wrong (action, target_kind) combo raises 400 — see Store
    implementation for the routing table.
    """
    try:
        action_id = await db.apply_moderation_action(
            report_id=body.report_id,
            target_kind=body.target_kind,
            target_id=body.target_id,
            action=body.action,
            reason=body.reason,
            actor_id=user["id"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rows = await db.list_moderation_actions_for_target(
        target_kind=body.target_kind, target_id=body.target_id, limit=1,
    )
    if not rows or rows[0]["id"] != action_id:
        # Defensive — list ordering should yield the row we just
        # inserted. If not, fall back to a minimal response.
        return ModerationActionOut(
            id=action_id, report_id=body.report_id,
            target_kind=body.target_kind, target_id=body.target_id,
            action=body.action, reason=body.reason,
            actor_id=user["id"], created_at=None,
        )
    return ModerationActionOut(**rows[0])


@admin_router.get(
    "/target/{target_kind}/{target_id}",
    response_model=list[ModerationActionOut],
)
async def list_actions_for_target(
    target_kind: ReportTargetKind,
    target_id:   int,
    limit:       int = Query(50, ge=1, le=200),
    db: Store = Depends(get_store),
):
    rows = await db.list_moderation_actions_for_target(
        target_kind=target_kind, target_id=target_id, limit=limit,
    )
    return [ModerationActionOut(**r) for r in rows]
