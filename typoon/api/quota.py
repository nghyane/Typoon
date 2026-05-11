"""Per-user translation quota — beta rate limit.

A "chapter slot" is consumed each time a user triggers an LLM-costing
action (draft create or render create for a translation). Cache hits
do NOT consume — quota tracks real cost only.

Counters are time-windowed over `chapter_consumes` rows (last hour,
last day). Admins bypass.

Routes call `enforce_chapter_quota(user, db, cfg, count=N)` BEFORE
enqueueing the LLM-costing work, then `record_consume(...)` AFTER each
successful spawn. Splitting the two lets a partial batch fail
explicitly instead of silently overshooting.
"""

from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from typoon.config import AuthConfig, RateLimitConfig
from typoon.storage import Store


_HOUR = 3600
_DAY  = 86400


def is_admin(user: dict, auth: AuthConfig) -> bool:
    rid = auth.admin_role_id
    return bool(rid) and rid in (user.get("roles") or [])


async def get_quota_snapshot(
    user: dict, db: Store, cfg: RateLimitConfig, auth: AuthConfig,
) -> dict:
    """Shape consumed by GET /api/me/quota and the SPA sidebar widget.

    Concurrency limits (in_flight) dropped in the material refactor —
    the old implementation joined through `projects.owner_id` which no
    longer exists. Hour + day windows are the meaningful gate.
    """
    admin = is_admin(user, auth)
    used_hour = await db.count_user_consumes_since(user["id"], _HOUR)
    used_day  = await db.count_user_consumes_since(user["id"], _DAY)
    return {
        "is_admin":       admin,
        "limit_hour":     cfg.chapters_per_hour,
        "used_hour":      used_hour,
        "remaining_hour": max(0, cfg.chapters_per_hour - used_hour),
        "limit_day":      cfg.chapters_per_day,
        "used_day":       used_day,
        "remaining_day":  max(0, cfg.chapters_per_day - used_day),
    }


async def enforce_chapter_quota(
    user: dict, db: Store, cfg: RateLimitConfig, auth: AuthConfig,
    count: int = 1,
) -> None:
    """Raise 429 if `count` more LLM events would exceed any window.

    Admin bypass. `count` is the number of slots the caller is about
    to consume (always 1 except for batch operations).
    """
    if is_admin(user, auth):
        return
    if count <= 0:
        return

    used_hour = await db.count_user_consumes_since(user["id"], _HOUR)
    if used_hour + count > cfg.chapters_per_hour:
        raise HTTPException(
            429,
            f"Đã dùng {used_hour}/{cfg.chapters_per_hour} chương "
            f"trong giờ này. Thử lại sau ít phút.",
            headers={"Retry-After": str(_HOUR)},
        )

    used_day = await db.count_user_consumes_since(user["id"], _DAY)
    if used_day + count > cfg.chapters_per_day:
        raise HTTPException(
            429,
            f"Đã dùng {used_day}/{cfg.chapters_per_day} chương hôm nay. "
            f"Quota reset mỗi 24h.",
            headers={"Retry-After": str(_DAY)},
        )


async def record_consume(
    user: dict, db: Store, auth: AuthConfig,
    *,
    translation_id: int,
    kind: Literal["draft_create", "render_create"],
) -> None:
    """Persist a consume row. Admins skipped (they bypass enforcement
    above; logging admin runs would distort user-facing counters)."""
    if is_admin(user, auth):
        return
    await db.record_chapter_consume(
        user_id=user["id"], translation_id=translation_id, kind=kind,
    )
