"""Per-user chapter quota — beta rate limit.

A "chapter slot" is consumed each time a user triggers an action that
will spend LLM tokens on a chapter: upload+start, manual /start, /redo.
Free reads (list, fetch) and idle uploads do not count.

Counters are time-windowed over `chapter_consumes` rows (last hour,
last day) plus a live-task count for concurrency. Admins bypass.

Routes call `enforce_chapter_quota(user, db, cfg, count=N)` BEFORE
enqueueing, then `record_chapter_consume(...)` AFTER each successful
enqueue. Splitting the two lets a partial batch (e.g. /chapters/start
with 10 ids when only 7 fit in the remaining hourly window) fail
explicitly instead of silently overshooting.
"""

from __future__ import annotations

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
    """Shape consumed by GET /api/me/quota and the SPA sidebar widget."""
    admin = is_admin(user, auth)
    used_hour    = await db.count_user_consumes_since(user["id"], _HOUR)
    used_day     = await db.count_user_consumes_since(user["id"], _DAY)
    in_flight    = await db.count_user_in_flight_chapters(user["id"])
    return {
        "is_admin":             admin,
        "limit_hour":           cfg.chapters_per_hour,
        "used_hour":            used_hour,
        "remaining_hour":       max(0, cfg.chapters_per_hour - used_hour),
        "limit_day":            cfg.chapters_per_day,
        "used_day":             used_day,
        "remaining_day":        max(0, cfg.chapters_per_day - used_day),
        "limit_concurrent":     cfg.concurrent_chapters,
        "in_flight":            in_flight,
        "remaining_concurrent": max(0, cfg.concurrent_chapters - in_flight),
    }


async def enforce_chapter_quota(
    user: dict, db: Store, cfg: RateLimitConfig, auth: AuthConfig,
    count: int = 1,
) -> None:
    """Raise 429 if `count` more chapters would exceed any window.

    Admin bypass. `count` is the number of slots the caller is about
    to consume in this request (always 1 except for batch /start).
    """
    if is_admin(user, auth):
        return
    if count <= 0:
        return

    in_flight = await db.count_user_in_flight_chapters(user["id"])
    if in_flight + count > cfg.concurrent_chapters:
        raise HTTPException(
            429,
            f"Đang xử lý {in_flight} chương; giới hạn đồng thời "
            f"là {cfg.concurrent_chapters}. Đợi xong rồi thử lại.",
        )

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


async def record_chapter_consume(
    user: dict, db: Store, auth: AuthConfig,
    chapter_id: int, project_id: int,
) -> None:
    """Persist a consume row. Admins skipped (they bypass enforcement
    above; logging admin runs would distort user-facing counters)."""
    if is_admin(user, auth):
        return
    await db.record_chapter_consume(user["id"], chapter_id, project_id)
