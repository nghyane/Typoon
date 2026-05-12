"""Tests for reports + moderation_actions.

Postgres-only. Skipped unless TEST_DATABASE_URL is set.

The path under test:
    1. user submits a report
    2. admin moves it to reviewing
    3. admin acts: takedown a draft
    4. admin restores the draft
    5. admin marks the report resolved

Each step writes to one of the two tables; we read back to assert.
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


async def _bootstrap_chapter_and_draft(store: PostgresStore) -> tuple[int, int, int, int]:
    """Create reporter, draft owner, material+chapter, and a real
    translation_drafts row so takedown/restore can flip its column.
    Returns (reporter_id, owner_id, chapter_id, draft_id)."""
    reporter = await store.upsert_user_from_identity(
        provider="test", external_id=f"r-{uuid.uuid4().hex[:8]}",
        display_name="Reporter",
    )
    owner = await store.upsert_user_from_identity(
        provider="test", external_id=f"o-{uuid.uuid4().hex[:8]}",
        display_name="Owner",
    )
    mid = await store.get_or_create_source_material(
        source="test",
        upstream_ref=f"m-{uuid.uuid4().hex[:8]}",
        title="X", languages=["ko"], imported_by=owner["id"],
    )
    cid = await store.create_chapter(material_id=mid, number="1")
    draft_id = await store.create_draft(
        chapter_id=cid, source_lang="ko", target_lang="vi",
        glossary_fp="00", llm_model="test/echo",
        created_by=owner["id"],
        visibility="guild", scope_guild_id="g-1",
    )
    return reporter["id"], owner["id"], cid, draft_id


@pytest.mark.asyncio
async def test_report_intake_does_not_touch_target():
    store = await PostgresStore.open(TEST_DSN)
    try:
        reporter_id, _, _, draft_id = await _bootstrap_chapter_and_draft(store)
        rid = await store.submit_report(
            reporter_id=reporter_id,
            reporter_label="Reporter",
            target_kind="draft", target_id=draft_id,
            scope_guild_id="g-1",
            kind="dmca", reason="not licensed",
        )
        # Status is open, target untouched.
        row = await store.get_report(rid)
        assert row is not None
        assert row["status"] == "open"
        draft = await store.get_draft(draft_id)
        assert draft is not None
        assert draft.get("takedown_at") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_takedown_then_restore_via_actions():
    store = await PostgresStore.open(TEST_DSN)
    try:
        reporter_id, owner_id, _, draft_id = await _bootstrap_chapter_and_draft(store)
        rid = await store.submit_report(
            reporter_id=reporter_id, reporter_label="Reporter",
            target_kind="draft", target_id=draft_id,
            scope_guild_id="g-1",
            kind="dmca", reason="not licensed",
        )

        # Admin moves to reviewing → status reflects.
        await store.update_report_status(
            rid, status="reviewing", resolver_id=owner_id,
        )
        assert (await store.get_report(rid))["status"] == "reviewing"

        # Takedown action — draft.takedown_at must be set.
        td_id = await store.apply_moderation_action(
            report_id=rid, target_kind="draft", target_id=draft_id,
            action="takedown", reason="DMCA notice 2026-05",
            actor_id=owner_id,
        )
        assert td_id > 0
        draft = await store.get_draft(draft_id)
        assert draft is not None
        assert draft.get("takedown_at") is not None
        assert draft.get("takedown_reason") == "DMCA notice 2026-05"

        # Restore — column clears.
        await store.apply_moderation_action(
            report_id=rid, target_kind="draft", target_id=draft_id,
            action="restore", reason="counter-notice accepted",
            actor_id=owner_id,
        )
        draft = await store.get_draft(draft_id)
        assert draft is not None
        assert draft.get("takedown_at") is None
        assert draft.get("takedown_reason") is None

        # Both actions are in the audit log.
        actions = await store.list_moderation_actions_for_target(
            target_kind="draft", target_id=draft_id, limit=10,
        )
        labels = [a["action"] for a in actions]
        assert labels[0] == "restore"
        assert labels[1] == "takedown"

        # Close out the report.
        await store.update_report_status(
            rid, status="resolved", resolver_id=owner_id,
        )
        final = await store.get_report(rid)
        assert final["status"] == "resolved"
        assert final["resolved_by"] == owner_id
        assert final["resolved_at"] is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_takedown_on_material_rejected():
    """`takedown` only applies to draft/translation. Trying it on
    material/chapter must raise — admin must use `delete` instead."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        owner = await store.upsert_user_from_identity(
            provider="test", external_id=f"x-{uuid.uuid4().hex[:8]}",
            display_name="X",
        )
        mid = await store.get_or_create_source_material(
            source="test", upstream_ref=f"m-{uuid.uuid4().hex[:8]}",
            title="X", languages=["ko"], imported_by=owner["id"],
        )
        with pytest.raises(ValueError):
            await store.apply_moderation_action(
                report_id=None, target_kind="material", target_id=mid,
                action="takedown", reason="no", actor_id=owner["id"],
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_proactive_action_without_report():
    """Admin can act without a triggering report — `report_id=None`."""
    store = await PostgresStore.open(TEST_DSN)
    try:
        _, owner_id, _, draft_id = await _bootstrap_chapter_and_draft(store)
        td_id = await store.apply_moderation_action(
            report_id=None, target_kind="draft", target_id=draft_id,
            action="takedown", reason="bulk takedown — proactive",
            actor_id=owner_id,
        )
        assert td_id > 0
        rows = await store.list_moderation_actions_for_target(
            target_kind="draft", target_id=draft_id, limit=1,
        )
        assert rows[0]["report_id"] is None
        assert rows[0]["action"] == "takedown"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_reports_filtered_by_status():
    store = await PostgresStore.open(TEST_DSN)
    try:
        reporter_id, owner_id, _, draft_id = await _bootstrap_chapter_and_draft(store)
        r1 = await store.submit_report(
            reporter_id=reporter_id, reporter_label="R",
            target_kind="draft", target_id=draft_id,
            scope_guild_id=None, kind="dmca", reason="a",
        )
        r2 = await store.submit_report(
            reporter_id=reporter_id, reporter_label="R",
            target_kind="draft", target_id=draft_id,
            scope_guild_id=None, kind="abuse", reason="b",
        )
        await store.update_report_status(
            r1, status="dismissed", resolver_id=owner_id,
        )
        open_ids      = {r["id"] for r in await store.list_reports(status="open",      limit=500)}
        dismissed_ids = {r["id"] for r in await store.list_reports(status="dismissed", limit=500)}
        assert r2 in open_ids
        assert r1 not in open_ids
        assert r1 in dismissed_ids
        assert r2 not in dismissed_ids
    finally:
        await store.close()
