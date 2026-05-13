"""Tests for translator_memory + translator_memory_briefs.

Postgres-only. Skipped unless TEST_DATABASE_URL points at a throwaway
database. Each test creates a fresh user + material so rows isolate
naturally; CASCADE on material/user delete cleans the rest.

Quick local run:
    createdb -O typoon typoon_test
    TEST_DATABASE_URL=postgresql://typoon:typoon@localhost:5432/typoon_test \\
        pytest tests/test_translator_memory.py
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


async def _bootstrap_user_material(store: PostgresStore) -> tuple[int, int]:
    """Create a throwaway user + source-backed material the tests can
    hang memory rows off. Returns (user_id, material_id)."""
    user = await store.upsert_user_from_identity(
        provider="test",
        external_id=f"u-{uuid.uuid4().hex[:8]}",
        display_name="Tester",
    )
    mat_id = await store.get_or_create_source_material(
        source="test",
        upstream_ref=f"m-{uuid.uuid4().hex[:8]}",
        title="Test Manga",
        languages=["ko"],
        imported_by=user["id"],
    )
    return user["id"], mat_id


@pytest.mark.asyncio
async def test_upsert_and_get_memory():
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid, mid = await _bootstrap_user_material(store)
        row = await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
            characters=[{"name": "Naruto", "pronouns": {"self": "tôi", "other": "cậu"}}],
            glossary=[{"source_term": "うずまき", "target_term": "Uzumaki"}],
        )
        assert row["source_lang"] == "ko"
        assert row["target_lang"] == "vi"
        assert row["characters"][0]["name"] == "Naruto"
        assert row["world"]    == {}
        assert row["style"]    == {}
        assert row["style_refs"] == []

        fetched = await store.get_translator_memory(
            user_id=uid, material_id=mid, target_lang="vi",
        )
        assert fetched is not None
        assert fetched["glossary"][0]["target_term"] == "Uzumaki"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_partial_update_preserves_other_cards():
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid, mid = await _bootstrap_user_material(store)
        await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
            characters=[{"name": "A"}],
            world={"setting": "Konoha"},
        )
        # Update only style — characters/world must stay.
        await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
            style={"tone": "casual"},
        )
        row = await store.get_translator_memory(
            user_id=uid, material_id=mid, target_lang="vi",
        )
        assert row is not None
        assert row["characters"] == [{"name": "A"}]
        assert row["world"]      == {"setting": "Konoha"}
        assert row["style"]      == {"tone": "casual"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_brief_sliding_window():
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid, mid = await _bootstrap_user_material(store)
        mem = await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
        )
        # Three chapters at increasing positions.
        c1 = await store.create_chapter(
            material_id=mid, number_norm="1",
        )
        c2 = await store.create_chapter(
            material_id=mid, number_norm="2",
        )
        c3 = await store.create_chapter(
            material_id=mid, number_norm="3",
        )
        for cid, summary in ((c1, "Ch1"), (c2, "Ch2"), (c3, "Ch3")):
            await store.append_memory_brief(
                memory_id=mem["id"], chapter_id=cid,
                brief_json={"summary": summary},
                summary=summary,
            )

        # No filter → newest first, capped at limit.
        recent = await store.list_recent_memory_briefs(
            memory_id=mem["id"], limit=2,
        )
        assert [b["summary"] for b in recent] == ["Ch3", "Ch2"]

        # `before_chapter_id=c3` → only c1, c2 eligible (strict <).
        window = await store.list_recent_memory_briefs(
            memory_id=mem["id"], before_chapter_id=c3, limit=5,
        )
        assert [b["summary"] for b in window] == ["Ch2", "Ch1"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_brief_upsert_overwrites():
    """Repeated translates of the same chapter overwrite, never duplicate."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid, mid = await _bootstrap_user_material(store)
        mem = await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
        )
        cid = await store.create_chapter(
            material_id=mid, number_norm="1",
        )
        await store.append_memory_brief(
            memory_id=mem["id"], chapter_id=cid,
            brief_json={"summary": "first"}, summary="first",
        )
        await store.append_memory_brief(
            memory_id=mem["id"], chapter_id=cid,
            brief_json={"summary": "second"}, summary="second",
        )
        rows = await store.list_recent_memory_briefs(
            memory_id=mem["id"], limit=10,
        )
        assert len(rows) == 1
        assert rows[0]["summary"] == "second"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_memory_cascades_briefs():
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid, mid = await _bootstrap_user_material(store)
        mem = await store.upsert_translator_memory(
            user_id=uid, material_id=mid,
            source_lang="ko", target_lang="vi",
        )
        cid = await store.create_chapter(
            material_id=mid, number_norm="1",
        )
        await store.append_memory_brief(
            memory_id=mem["id"], chapter_id=cid,
            brief_json={"summary": "x"}, summary="x",
        )
        ok = await store.delete_translator_memory(
            user_id=uid, material_id=mid, target_lang="vi",
        )
        assert ok is True
        assert await store.get_translator_memory(
            user_id=uid, material_id=mid, target_lang="vi",
        ) is None
        # No orphan briefs (cascade).
        rows = await store.list_recent_memory_briefs(
            memory_id=mem["id"], limit=10,
        )
        assert rows == []
    finally:
        await store.close()
