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


# ── Work payload (GET /api/work/{id} backing store calls) ────────────


@pytest.mark.asyncio
async def test_work_payload_lists_siblings_and_translations():
    """`list_materials_for_work` + `list_work_chapters_with_translations`
    together drive the `/work/{id}` endpoint. Together they must:

      - list every material attached to the Work (oldest-first),
      - list every `work_chapter` the community has touched,
      - attach every shared translation to the right chapter, including
        translations spawned from a *different* sibling material than
        the one a hypothetical viewer is currently looking at.
    """
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
        work_id = int(ma["work_id"])
        assert work_id == int(mb["work_id"])

        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40", label="Chương 40",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        assert ca is not None
        draft_id = await store.create_draft(
            chapter_id=c_a,
            source_lang="ko", target_lang="vi",
            glossary_fp="fp" + uuid.uuid4().hex[:14],
            llm_model="test-model",
            created_by=uid_a,
        )
        trans_id = await store.get_or_create_translation(
            work_chapter_id=int(ca["work_chapter_id"]),
            owner_id=uid_a,
            target_lang="vi",
            draft_id=draft_id,
            shared=True,
        )

        # Sibling materials list, oldest first.
        mats = await store.list_materials_for_work(work_id)
        assert [m["id"] for m in mats] == [ma["id"], mb["id"]]

        # Viewer B sees user A's translation under the touched chapter.
        chapters = await store.list_work_chapters_with_translations(
            work_id, viewer_id=uid_b,
        )
        assert len(chapters) == 1
        ch = chapters[0]
        assert ch["id"] == int(ca["work_chapter_id"])
        assert ch["number_norm"] == "40"
        assert len(ch["translations"]) == 1
        tr = ch["translations"][0]
        assert tr["id"] == trans_id
        assert tr["target_lang"] == "vi"
        # draft_material_id surfaces the source whose pixels back the
        # render — material A here, even though we're viewing as B.
        assert tr["draft_material_id"] == ma["id"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_work_payload_filters_unshared_for_other_viewers():
    """A private (`shared=False`) translation is visible to its owner
    via `list_work_chapters_with_translations` but hidden from any
    other viewer."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        uid_owner   = await _user(store)
        uid_visitor = await _user(store)
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": f"mdex-{uuid.uuid4().hex[:8]}"},
            imported_by=uid_owner,
        )
        c_a = await store.create_chapter(
            material_id=ma["id"], number_norm="40",
            upstream_url=f"https://a.example/{uuid.uuid4().hex}",
        )
        ca = await store.get_chapter(c_a)
        assert ca is not None
        draft_id = await store.create_draft(
            chapter_id=c_a,
            source_lang="ko", target_lang="vi",
            glossary_fp="fp" + uuid.uuid4().hex[:14],
            llm_model="test-model",
            created_by=uid_owner,
        )
        await store.get_or_create_translation(
            work_chapter_id=int(ca["work_chapter_id"]),
            owner_id=uid_owner,
            target_lang="vi",
            draft_id=draft_id,
            shared=False,  # private
        )

        work_id = int(ma["work_id"])
        owner_view = await store.list_work_chapters_with_translations(
            work_id, viewer_id=uid_owner,
        )
        visitor_view = await store.list_work_chapters_with_translations(
            work_id, viewer_id=uid_visitor,
        )
        assert len(owner_view[0]["translations"]) == 1
        assert len(visitor_view[0]["translations"]) == 0
    finally:
        await store.close()


# ── Community link-vote merge ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_link_vote_below_threshold_stores_but_does_not_merge():
    """Two distinct +1 votes (under the threshold of 3) leave the
    two materials in separate Works. Suggestion appears in the list."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        v1 = await _user(store)
        v2 = await _user(store)
        ma = await _material(
            store, source="src-a", cross_refs=None, imported_by=v1,
        )
        mb = await _material(
            store, source="src-b", cross_refs=None, imported_by=v1,
        )
        assert ma["work_id"] != mb["work_id"]

        r1 = await store.cast_link_vote_with_merge(
            voter_id=v1,
            material_a_id=ma["id"], material_b_id=mb["id"],
            vote=1, threshold=3,
        )
        assert r1 == {
            "vote": 1, "score": 1, "merged": False,
            "canonical_work_id": None, "blocked_reason": None,
        }
        r2 = await store.cast_link_vote_with_merge(
            voter_id=v2,
            material_a_id=ma["id"], material_b_id=mb["id"],
            vote=1, threshold=3,
        )
        assert r2["score"] == 2
        assert r2["merged"] is False

        # Re-fetch — Works still separate, vote score reflected in
        # the suggestion list of either Work.
        ma2 = await store.get_material(ma["id"])
        mb2 = await store.get_material(mb["id"])
        assert ma2["work_id"] != mb2["work_id"]

        sug = await store.list_work_link_suggestions(work_id=int(ma2["work_id"]))
        assert len(sug) == 1
        assert sug[0]["candidate_material_id"] == mb["id"]
        assert sug[0]["score"] == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_link_vote_crosses_threshold_merges_inline():
    """Three distinct +1 votes pass the threshold → the doomed Work
    dissolves into the older canonical Work in the same call."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        v1, v2, v3 = (
            await _user(store), await _user(store), await _user(store),
        )
        ma = await _material(
            store, source="src-a", cross_refs=None, imported_by=v1,
        )
        mb = await _material(
            store, source="src-b", cross_refs=None, imported_by=v1,
        )
        # ma is older → its work is canonical.
        canonical = min(int(ma["work_id"]), int(mb["work_id"]))

        for v in (v1, v2, v3):
            r = await store.cast_link_vote_with_merge(
                voter_id=v,
                material_a_id=ma["id"], material_b_id=mb["id"],
                vote=1, threshold=3,
            )
        assert r["merged"] is True
        assert r["canonical_work_id"] == canonical

        # Both materials now share the canonical Work.
        ma2 = await store.get_material(ma["id"])
        mb2 = await store.get_material(mb["id"])
        assert int(ma2["work_id"]) == canonical
        assert int(mb2["work_id"]) == canonical
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_link_vote_blocked_when_cross_refs_conflict():
    """Two Works each carrying a different `mdex_uuid` are an identity
    collision: even at 3+ positive votes the merge MUST be refused.
    """
    store = await PostgresStore.open(TEST_DSN)
    try:
        v1, v2, v3 = (
            await _user(store), await _user(store), await _user(store),
        )
        ma = await _material(
            store, source="src-a",
            cross_refs={"mdex_uuid": "MDEX-AAA"},
            imported_by=v1,
        )
        mb = await _material(
            store, source="src-b",
            cross_refs={"mdex_uuid": "MDEX-BBB"},
            imported_by=v1,
        )
        for v in (v1, v2, v3):
            r = await store.cast_link_vote_with_merge(
                voter_id=v,
                material_a_id=ma["id"], material_b_id=mb["id"],
                vote=1, threshold=3,
            )
        assert r["merged"] is False
        assert r["blocked_reason"] == "cross_refs_conflict"

        ma2 = await store.get_material(ma["id"])
        mb2 = await store.get_material(mb["id"])
        assert ma2["work_id"] != mb2["work_id"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_merge_remaps_chapters_translations_and_history():
    """When two Works merge, every dependent row on the doomed side
    moves to the canonical side. Chapters that already exist on
    canonical at the same `number_norm` collapse onto canonical's
    row; their translations + reading_history follow."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        v1, v2, v3 = (
            await _user(store), await _user(store), await _user(store),
        )
        ma = await _material(
            store, source="src-a", cross_refs=None, imported_by=v1,
        )
        mb = await _material(
            store, source="src-b", cross_refs=None, imported_by=v1,
        )

        # Each material gets a chapter at number_norm "40". Pre-merge
        # they live on different work_chapters.
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
        assert ca["work_chapter_id"] != cb["work_chapter_id"]

        # Translation on the b side that must survive the merge.
        draft_id = await store.create_draft(
            chapter_id=c_b,
            source_lang="ko", target_lang="vi",
            glossary_fp="fp" + uuid.uuid4().hex[:14],
            llm_model="test-model",
            created_by=v1,
        )
        trans_id = await store.get_or_create_translation(
            work_chapter_id=int(cb["work_chapter_id"]),
            owner_id=v1,
            target_lang="vi",
            draft_id=draft_id,
            shared=True,
        )
        # Reading history on the b side too.
        await store.record_reading(
            user_id=v1,
            work_chapter_id=int(cb["work_chapter_id"]),
            last_material_id=mb["id"],
            translation_id=trans_id,
        )

        for v in (v1, v2, v3):
            r = await store.cast_link_vote_with_merge(
                voter_id=v,
                material_a_id=ma["id"], material_b_id=mb["id"],
                vote=1, threshold=3,
            )
        assert r["merged"] is True
        canonical = int(r["canonical_work_id"])

        # After merge: canonical Work has exactly one work_chapter
        # at number_norm "40"; b's translation + history now point
        # at it.
        chapters = await store.list_work_chapters_with_translations(
            canonical, viewer_id=v1,
        )
        assert len(chapters) == 1
        merged_wc = chapters[0]["id"]
        assert chapters[0]["number_norm"] == "40"
        trs = chapters[0]["translations"]
        assert len(trs) == 1
        assert trs[0]["id"] == trans_id

        # Chapter rows from both materials point to the merged WC.
        ca2 = await store.get_chapter(c_a)
        cb2 = await store.get_chapter(c_b)
        assert int(ca2["work_chapter_id"]) == merged_wc
        assert int(cb2["work_chapter_id"]) == merged_wc

        # Reading history followed.
        recent = await store.list_recent_reads(user_id=v1, limit=10)
        assert len(recent) == 1
        assert recent[0]["work_chapter_id"] == merged_wc
    finally:
        await store.close()
