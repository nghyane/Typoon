"""Tests for ChapterBrief storage."""

from __future__ import annotations

import pytest

from typoon.storage.sqlite import SqliteStore


@pytest.mark.asyncio
async def test_save_and_get_recent_chapter_briefs():
    store = await SqliteStore.open_memory()
    try:
        pid  = await store.get_or_create_project("test", "T", "ko", "vi")
        cid1 = await store.get_or_create_chapter(pid, 1.0)
        cid2 = await store.get_or_create_chapter(pid, 2.0)
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
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_chapter_cascades_brief():
    store = await SqliteStore.open_memory()
    try:
        pid = await store.get_or_create_project("test2", "T", "ko", "vi")
        cid = await store.get_or_create_chapter(pid, 1.0)
        await store.save_chapter_brief(cid, {"summary": "gone"})
        # Cascade delete via chapters ON DELETE CASCADE
        await store._db.execute("DELETE FROM chapters WHERE id=?", (cid,))
        await store._db.commit()
        assert await store.get_chapter_brief(cid) is None
    finally:
        await store.close()
