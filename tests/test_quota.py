"""Quota enforcement contract — beta translation rate limit.

Pure logic test: stub `Store` for the two counter methods we read.
No Postgres needed; the SQL layer is exercised by integration runs.

After the material refactor `chapter_consumes` rows are keyed by
`(user_id, translation_id, kind)`; the concurrency limit went away
with `projects.owner_id`. This test covers the hour/day windows,
admin bypass, batched counts, and the snapshot shape consumed by
`GET /api/me/quota`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from typoon.api.quota import (
    enforce_chapter_quota, get_quota_snapshot, is_admin, record_consume,
)
from typoon.config import AuthConfig, RateLimitConfig


class StubStore:
    """Minimal Store stand-in covering the two methods quota.py calls.

    `since_seconds → count` maps a window length to the number of
    consume rows the stub should report; defaults bucket hourly vs
    daily on the 3600s boundary, matching real call sites.
    """

    def __init__(self, hour: int, day: int) -> None:
        self._hour = hour
        self._day  = day
        self.recorded: list[tuple[int, int, str]] = []

    async def count_user_consumes_since(
        self, user_id: int, seconds: int,
    ) -> int:
        return self._hour if seconds <= 3600 else self._day

    async def record_chapter_consume(
        self, *, user_id: int, translation_id: int, kind: str,
    ) -> None:
        self.recorded.append((user_id, translation_id, kind))


def _user(roles: list[str] | None = None) -> dict:
    return {"id": 1, "roles": roles or []}


def _auth(admin_role: str = "") -> AuthConfig:
    cfg = AuthConfig()
    cfg.admin_role_id = admin_role
    return cfg


@pytest.mark.asyncio
async def test_enforce_passes_under_all_limits():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=50)
    store = StubStore(hour=3, day=12)
    await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)


@pytest.mark.asyncio
async def test_enforce_blocks_on_hourly_cap():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=100)
    store = StubStore(hour=10, day=10)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "3600"


@pytest.mark.asyncio
async def test_enforce_blocks_on_daily_cap():
    rl = RateLimitConfig(chapters_per_hour=100, chapters_per_day=50)
    store = StubStore(hour=0, day=50)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "86400"


@pytest.mark.asyncio
async def test_enforce_count_n_respects_remaining_window():
    """Batch of 5 against an hourly window with only 3 slots left → 429."""
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=100)
    store = StubStore(hour=7, day=7)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=5)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_enforce_zero_count_is_noop():
    """A zero-slot enforce call must not raise even at the cap."""
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=50)
    store = StubStore(hour=10, day=50)
    await enforce_chapter_quota(_user(), store, rl, _auth(), count=0)


@pytest.mark.asyncio
async def test_admin_bypasses_all_caps():
    rl = RateLimitConfig(chapters_per_hour=1, chapters_per_day=1)
    store = StubStore(hour=999, day=999)
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user(roles=["ADMIN_ROLE"])
    await enforce_chapter_quota(user, store, rl, auth, count=10)


@pytest.mark.asyncio
async def test_record_consume_skips_admin():
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user(roles=["ADMIN_ROLE"])
    store = StubStore(hour=0, day=0)
    await record_consume(
        user, store, auth, translation_id=42, kind="draft_create",
    )
    assert store.recorded == []


@pytest.mark.asyncio
async def test_record_consume_logs_normal_user():
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user()
    store = StubStore(hour=0, day=0)
    await record_consume(
        user, store, auth, translation_id=42, kind="render_create",
    )
    assert store.recorded == [(1, 42, "render_create")]


@pytest.mark.asyncio
async def test_snapshot_shape_for_normal_user():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=50)
    store = StubStore(hour=2, day=8)
    snap = await get_quota_snapshot(_user(), store, rl, _auth())
    assert snap == {
        "is_admin":       False,
        "limit_hour":     10,
        "used_hour":      2,
        "remaining_hour": 8,
        "limit_day":      50,
        "used_day":       8,
        "remaining_day":  42,
    }


@pytest.mark.asyncio
async def test_snapshot_flags_admin():
    rl = RateLimitConfig()
    store = StubStore(hour=0, day=0)
    auth = _auth(admin_role="ADMIN")
    user = _user(roles=["ADMIN"])
    snap = await get_quota_snapshot(user, store, rl, auth)
    assert snap["is_admin"] is True


def test_is_admin_requires_role_id_configured():
    user = _user(roles=["SOMEROLE"])
    assert is_admin(user, _auth(admin_role="")) is False
    assert is_admin(user, _auth(admin_role="OTHERROLE")) is False
    assert is_admin(user, _auth(admin_role="SOMEROLE")) is True
