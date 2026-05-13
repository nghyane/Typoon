"""End-to-end tests for the Work / WorkChapter identity model.

Postgres-only. Same TEST_DATABASE_URL gate as the rest of the
storage tests. Each test bootstraps its own user + (material × N)
so rows isolate naturally; CASCADE on material/user delete cleans
the rest.

Covers the three Commit 1b promises:

  1. Auto-link by cross_refs — two source materials sharing a
     mdex_uuid collapse to one Work; disjoint cross_refs stay
     isolated; conflicting cross_refs (same namespace, different
     value) refuse to merge.
  2. Work_chapter materialise on demand — create_chapter inserts a
     work_chapters row for (work_id, number_norm) and reuses it on
     subsequent calls (cross-material) inside the same Work.
  3. Translation overlay crosses sources — a translation spawned
     from material A surfaces on `list_translations_for_chapters`
     for the sibling chapter of material B once they share a Work
     chapter.
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


async def _user(store: PostgresStore) -> int:
    row = await store.upsert_user_from_identity(
        provider="test",
        external_id=f"u-{uuid.uuid4().hex[:8]}",
        display_name="Tester",
    )
    return row["id"]


async def _material(
    store: PostgresStore,
    *,
    source: str,
    cross_refs: dict | None = None,
    imported_by: int | None = None,
) -> dict:
    """Create a fresh source-backed material and return its row."""
    mid = await store.get_or_create_source_material(
        source=source,
        upstream_ref=f"m-{uuid.uuid4().hex[:8]}",
        title="Test Manga",
        languages=["ko"],
        cross_refs=cross_refs,
        imported_by=imported_by,
    )
    row = await store.get_material(mid)
    assert row is not None
    return row


# ── auto-link by cross_refs ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_link_matches_on_overlapping_cross_ref():
    """Two materials sharing a mdex_uuid attach to the same Work."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        ref = f"mdex-{uuid.uuid4().hex[:8]}"
        m1 = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": ref},
            imported_by=uid,
        )
        m2 = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": ref, "anilist": 12345},
            imported_by=uid,
        )
        assert m1["work_id"] == m2["work_id"]
        # The second material's extra namespace is merged in.
        work = await store.get_work(m1["work_id"])
        assert work is not None
        assert work["cross_refs"]["mdex_uuid"] == ref
        assert str(work["cross_refs"]["anilist"]) == "12345"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_auto_link_isolates_when_no_overlap():
    """Materials with disjoint (or absent) cross_refs each get their
    own Work — community vote merges later."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        m1 = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": f"mdex-{uuid.uuid4().hex[:8]}"},
            imported_by=uid,
        )
        m2 = await _material(
            store, source="src-b",
            cross_refs=None,
            imported_by=uid,
        )
        assert m1["work_id"] != m2["work_id"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_auto_link_blocks_on_conflicting_cross_ref():
    """Same namespace, different value → identity collision: do NOT
    attach to the existing Work; create a fresh isolated one."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        m1 = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": "MDEX-AAA"},
            imported_by=uid,
        )
        m2 = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": "MDEX-BBB"},
            imported_by=uid,
        )
        assert m1["work_id"] != m2["work_id"]
    finally:
        await store.close()


# ── work_chapter materialise on demand ──────────────────────────────


@pytest.mark.asyncio
async def test_create_chapter_reuses_work_chapter_across_materials():
    """Two materials of the same Work, each adding their own chapter
    with the same number_norm, must point at the SAME work_chapter."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        ref = f"mdex-{uuid.uuid4().hex[:8]}"
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": ref}, imported_by=uid,
        )
        mb = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": ref}, imported_by=uid,
        )
        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40", label="Chương 040",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        c_b = await store.create_chapter(
            material_id=mb["id"], number_norm="40", label="Chapter 40",
            upstream_url=f"https://b.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        cb = await store.get_chapter(c_b)
        assert ca is not None and cb is not None
        assert ca["work_chapter_id"] == cb["work_chapter_id"]
        # Server returns canonical key in `number_norm` (alias for the
        # dropped `chapters.number` column).
        assert ca["number_norm"] == "40"
        assert cb["number_norm"] == "40"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_create_chapter_isolates_when_works_differ():
    """Different Works → independent work_chapters even at the same
    number_norm."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": f"mdex-{uuid.uuid4().hex[:8]}"},
            imported_by=uid,
        )
        mb = await _material(
            store, source="src-b",
            cross_refs=None,  # isolated Work
            imported_by=uid,
        )
        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        c_b = await store.create_chapter(
            material_id=mb["id"], number_norm="40",
            upstream_url=f"https://b.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        cb = await store.get_chapter(c_b)
        assert ca is not None and cb is not None
        assert ca["work_chapter_id"] != cb["work_chapter_id"]
    finally:
        await store.close()


# ── translation surfaces cross-source ────────────────────────────────


@pytest.mark.asyncio
async def test_translation_overlay_crosses_sources():
    """User A spawns a translation from material A; user B opening
    material B (same Work) sees that translation on the matching
    chapter via list_translations_for_chapters."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid_a = await _user(store)
        uid_b = await _user(store)
        ref = f"mdex-{uuid.uuid4().hex[:8]}"
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": ref}, imported_by=uid_a,
        )
        mb = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": ref}, imported_by=uid_b,
        )
        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40", label="Chương 40",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        c_b = await store.create_chapter(
            material_id=mb["id"], number_norm="40", label="Chapter 40",
            upstream_url=f"https://b.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        assert ca is not None

        # User A spawns a draft on chapter A.
        draft_id = await store.create_draft(
            chapter_id=c_a,
            source_lang="ko", target_lang="vi",
            glossary_fp="fp" + uuid.uuid4().hex[:14],
            llm_model="test-model",
            created_by=uid_a,
        )
        # Translation row sits at Work-chapter scope.
        trans_id = await store.get_or_create_translation(
            work_chapter_id=int(ca["work_chapter_id"]),
            owner_id=uid_a,
            target_lang="vi",
            draft_id=draft_id,
            shared=True,
        )

        # Querying overlay for chapter B (different material) must
        # surface user A's translation — they share the work_chapter.
        overlay = await store.list_translations_for_chapters([c_b])
        rows = overlay.get(c_b, [])
        assert any(r["id"] == trans_id for r in rows), (
            f"translation {trans_id} should appear under chapter_b={c_b}; "
            f"got {rows}"
        )
    finally:
        await store.close()


# ── reading_history dedup by work_chapter ────────────────────────────


@pytest.mark.asyncio
async def test_reading_history_dedupes_per_work_chapter():
    """Reading chapter 40 from material A then from material B (same
    Work) collapses to ONE history row keyed on (user, work_chapter)."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid = await _user(store)
        ref = f"mdex-{uuid.uuid4().hex[:8]}"
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": ref}, imported_by=uid,
        )
        mb = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": ref}, imported_by=uid,
        )
        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        c_b = await store.create_chapter(
            material_id=mb["id"], number_norm="40",
            upstream_url=f"https://b.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        cb = await store.get_chapter(c_b)
        assert ca is not None and cb is not None
        wc_id = int(ca["work_chapter_id"])
        assert wc_id == int(cb["work_chapter_id"])

        await store.record_reading(
            user_id=uid, work_chapter_id=wc_id,
            last_material_id=ma["id"], translation_id=None,
        )
        await store.record_reading(
            user_id=uid, work_chapter_id=wc_id,
            last_material_id=mb["id"], translation_id=None,
        )
        recent = await store.list_recent_reads(user_id=uid, limit=10)
        # Both reads collapse to one entry; `material_id` reflects the
        # latest source the user actually opened.
        assert len(recent) == 1
        assert recent[0]["work_chapter_id"] == wc_id
        assert recent[0]["material_id"] == mb["id"]
    finally:
        await store.close()
