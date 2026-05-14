"""Backfill: re-link every material whose cross_refs already exist
but whose Work was created before the enrich-driven re-link landed.

Run once after deploying the `_relink_material_if_needed` change.
Idempotent — re-running it on already-merged data is a no-op.

Usage:
    python -m scripts.backfill_relink_materials

Reads DATABASE_URL from the standard env loader so it matches the
running API process.
"""

from __future__ import annotations

import asyncio
import logging

from typoon.config import load_config
from typoon.storage.postgres import PostgresStore


log = logging.getLogger("backfill.relink")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg, _paths = load_config()
    store = await PostgresStore.open(cfg.database_url)
    try:
        rows = await store._pool.fetch(   # type: ignore[attr-defined]
            "SELECT id, work_id FROM materials "
            "WHERE cross_refs IS NOT NULL "
            "  AND jsonb_typeof(cross_refs) = 'object' "
            "  AND cross_refs <> '{}'::jsonb "
            "ORDER BY id"
        )
        log.info("found %d materials with cross_refs to re-link", len(rows))

        moved = 0
        unchanged = 0
        for row in rows:
            mid = int(row["id"])
            old_wid = int(row["work_id"])
            async with store._pool.acquire() as conn, conn.transaction():   # type: ignore[attr-defined]
                await store._relink_material_if_needed(conn, mid)           # type: ignore[attr-defined]
            new_row = await store._pool.fetchrow(                            # type: ignore[attr-defined]
                "SELECT work_id FROM materials WHERE id=$1", mid,
            )
            new_wid = int(new_row["work_id"]) if new_row else old_wid
            if new_wid != old_wid:
                log.info("material %d: work %d → %d", mid, old_wid, new_wid)
                moved += 1
            else:
                unchanged += 1

        log.info("done: %d moved, %d unchanged", moved, unchanged)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
