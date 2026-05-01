"""Tests for ChapterBrief storage."""

from __future__ import annotations

import pytest

from typoon.storage.sqlite import SqliteStore


@pytest.mark.asyncio
async def test_save_and_get_recent_chapter_briefs():
    store = await SqliteStore.open_memory()
    try:
        pid = await store.add_project("test", "T")
        await store.add_chapter(pid, 1)
        await store.add_chapter(pid, 2)
        await store.save_chapter_brief(pid, 1, {
            "summary": "First",
            "facts": ["A meets B"],
            "glossary": {"A": "A"},
            "rules": ["casual"],
            "page_notes": {},
            "key_notes": {},
        })
        recent = await store.get_recent_chapter_briefs(pid, before_chapter=2, limit=3)
        assert len(recent) == 1
        assert recent[0]["brief"]["summary"] == "First"
        assert "A meets B" in recent[0]["facts_text"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_chapter_data_removes_brief():
    store = await SqliteStore.open_memory()
    try:
        pid = await store.add_project("test2", "T")
        await store.add_chapter(pid, 1)
        await store.save_chapter_brief(pid, 1, {"summary": "gone"})
        await store.delete_chapter_data(pid, 1)
        assert await store.get_chapter_brief(pid, 1) is None
    finally:
        await store.close()
