"""Tests for ChapterBrief storage.

Postgres-only (RFC-005). Skipped unless TEST_DATABASE_URL points at a
throwaway Postgres database. The schema is created idempotently via
`PostgresStore.open`; each test inserts under a unique slug and cleans
up via `delete_project` (cascade).

Quick local run:
    createdb -O typoon typoon_test  # one-time
    TEST_DATABASE_URL=postgresql://typoon:typoon@localhost:5432/typoon_test \\
        pytest tests/test_chapter_brief_store.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from typoon.storage.postgres import PostgresStore

TEST_DSN = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not TEST_DSN,
    reason="set TEST_DATABASE_URL=postgresql://… to run storage tests",
)


@pytest.mark.asyncio
async def test_save_and_get_recent_chapter_briefs():
    store = await PostgresStore.open(TEST_DSN)
    slug = f"test-{uuid.uuid4().hex[:8]}"
    try:
        pid  = await store.get_or_create_project(slug, "T", "ko", "vi")
        cid1 = await store.get_or_create_chapter(pid, 1.0)
        await store.get_or_create_chapter(pid, 2.0)
        await store.save_chapter_brief(cid1, {
            "summary": "First",
            "facts": ["A meets B"],
            "glossary": {"A": "A"},
            "rules": ["casual"],
        })
        recent = await store.get_recent_chapter_briefs(pid, before_chapter_idx=2.0, limit=3)
        assert len(recent) == 1
        assert recent[0]["brief"]["summary"] == "First"
        assert "A meets B" in recent[0]["facts_text"]
        await store.delete_project(pid)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_chapter_cascades_brief():
    store = await PostgresStore.open(TEST_DSN)
    slug = f"test-{uuid.uuid4().hex[:8]}"
    try:
        pid = await store.get_or_create_project(slug, "T", "ko", "vi")
        cid = await store.get_or_create_chapter(pid, 1.0)
        await store.save_chapter_brief(cid, {"summary": "gone"})
        await store.delete_chapter(cid)
        assert await store.get_chapter_brief(cid) is None
        await store.delete_project(pid)
    finally:
        await store.close()
