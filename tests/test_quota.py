"""Quota enforcement contract — beta chapter rate limit.

Pure logic test: stub `Store` for the three counter methods we read.
No Postgres needed; the SQL layer is exercised by integration runs.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from typoon.api.quota import (
    enforce_chapter_quota, get_quota_snapshot, is_admin,
    record_chapter_consume,
)
from typoon.config import AuthConfig, RateLimitConfig


class StubStore:
    def __init__(self, hour: int, day: int, in_flight: int):
        self._hour = hour
        self._day  = day
        self._in_flight = in_flight
        self.recorded: list[tuple[int, int, int]] = []

    async def count_user_consumes_since(self, user_id: int, seconds: int) -> int:
        return self._hour if seconds <= 3600 else self._day

    async def count_user_in_flight_chapters(self, user_id: int) -> int:
        return self._in_flight

    async def record_chapter_consume(
        self, user_id: int, chapter_id: int, project_id: int,
    ) -> None:
        self.recorded.append((user_id, chapter_id, project_id))


def _user(roles: list[str] | None = None) -> dict:
    return {"id": 1, "roles": roles or []}


def _auth(admin_role: str = "") -> AuthConfig:
    cfg = AuthConfig()
    cfg.admin_role_id = admin_role
    return cfg


@pytest.mark.asyncio
async def test_enforce_passes_under_all_limits():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=50, concurrent_chapters=3)
    store = StubStore(hour=3, day=12, in_flight=1)
    await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)


@pytest.mark.asyncio
async def test_enforce_blocks_on_concurrent_cap():
    rl = RateLimitConfig(chapters_per_hour=100, chapters_per_day=100, concurrent_chapters=3)
    store = StubStore(hour=0, day=0, in_flight=3)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)
    assert exc.value.status_code == 429
    assert "đồng thời" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_enforce_blocks_on_hourly_cap():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=100, concurrent_chapters=10)
    store = StubStore(hour=10, day=10, in_flight=0)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "3600"


@pytest.mark.asyncio
async def test_enforce_blocks_on_daily_cap():
    rl = RateLimitConfig(chapters_per_hour=100, chapters_per_day=50, concurrent_chapters=10)
    store = StubStore(hour=0, day=50, in_flight=0)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "86400"


@pytest.mark.asyncio
async def test_enforce_count_n_respects_remaining_window():
    """Batch of 5 against an hourly window with only 3 slots left → 429."""
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=100, concurrent_chapters=10)
    store = StubStore(hour=7, day=7, in_flight=0)
    with pytest.raises(HTTPException) as exc:
        await enforce_chapter_quota(_user(), store, rl, _auth(), count=5)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_admin_bypasses_all_caps():
    rl = RateLimitConfig(chapters_per_hour=1, chapters_per_day=1, concurrent_chapters=1)
    store = StubStore(hour=999, day=999, in_flight=999)
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user(roles=["ADMIN_ROLE"])
    # No raise.
    await enforce_chapter_quota(user, store, rl, auth, count=10)


@pytest.mark.asyncio
async def test_record_consume_skips_admin():
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user(roles=["ADMIN_ROLE"])
    store = StubStore(hour=0, day=0, in_flight=0)
    await record_chapter_consume(user, store, auth, chapter_id=42, project_id=7)
    assert store.recorded == []


@pytest.mark.asyncio
async def test_record_consume_logs_normal_user():
    auth = _auth(admin_role="ADMIN_ROLE")
    user = _user()
    store = StubStore(hour=0, day=0, in_flight=0)
    await record_chapter_consume(user, store, auth, chapter_id=42, project_id=7)
    assert store.recorded == [(1, 42, 7)]


@pytest.mark.asyncio
async def test_snapshot_shape_for_normal_user():
    rl = RateLimitConfig(chapters_per_hour=10, chapters_per_day=50, concurrent_chapters=3)
    store = StubStore(hour=2, day=8, in_flight=1)
    snap = await get_quota_snapshot(_user(), store, rl, _auth())
    assert snap == {
        "is_admin":             False,
        "limit_hour":           10,
        "used_hour":            2,
        "remaining_hour":       8,
        "limit_day":            50,
        "used_day":             8,
        "remaining_day":        42,
        "limit_concurrent":     3,
        "in_flight":            1,
        "remaining_concurrent": 2,
    }


@pytest.mark.asyncio
async def test_snapshot_flags_admin():
    rl = RateLimitConfig()
    store = StubStore(hour=0, day=0, in_flight=0)
    auth = _auth(admin_role="ADMIN")
    user = _user(roles=["ADMIN"])
    snap = await get_quota_snapshot(user, store, rl, auth)
    assert snap["is_admin"] is True


def test_is_admin_requires_role_id_configured():
    user = _user(roles=["SOMEROLE"])
    assert is_admin(user, _auth(admin_role="")) is False
    assert is_admin(user, _auth(admin_role="OTHERROLE")) is False
    assert is_admin(user, _auth(admin_role="SOMEROLE")) is True
