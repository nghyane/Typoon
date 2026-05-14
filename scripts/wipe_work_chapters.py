"""Wipe a work's chapter spine + dependent rows for a clean re-fetch.

Use case: after fixing the chapter-number normaliser for a source
(e.g. HappyMH `@order` → label-based), existing `work_chapters` rows
carry the OLD `number_norm` keys. The frontend re-fetch produces NEW
rows with correct keys; the old ones stay around as duplicates the
chapter list can't merge.

This script removes the offending data so the next manifest fetch
rebuilds the spine clean. Translations / drafts / archives are
cascade-deleted by Postgres FKs — accept the loss in exchange for a
clean state. Run in dev only; production data needs Option B
(in-place re-normalisation) which we haven't built.

Usage:
    python -m scripts.wipe_work_chapters <work_id>

Lists what will be deleted and asks for confirmation unless `-y`
is passed.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from typoon.config import load_config
from typoon.storage.postgres import PostgresStore


log = logging.getLogger("wipe.work")


async def main(work_id: int, *, assume_yes: bool) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg, _paths = load_config()
    store = await PostgresStore.open(cfg.database_url)
    try:
        counts = await _count(store, work_id)
        if counts["work_chapters"] == 0:
            log.info("work %d has no work_chapters — nothing to wipe", work_id)
            return 0
        log.info("work %d will lose:", work_id)
        for k, v in counts.items():
            log.info("  %-20s %d", k, v)

        if not assume_yes:
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans != "y":
                log.info("aborted")
                return 1

        # Order matters: chapters.work_chapter_id is NO ACTION (not
        # CASCADE) — work_chapter is a logical key, chapter is a
        # physical per-material row that owns its own lifecycle.
        # Translations/reading_history sit between them via FK.
        # Wipe leaf rows first, then the chapters, then the spine.
        async with store._pool.acquire() as conn, conn.transaction():   # type: ignore[attr-defined]
            await conn.execute(
                "DELETE FROM reading_history rh USING work_chapters wc "
                "WHERE rh.work_chapter_id = wc.id AND wc.work_id=$1",
                work_id,
            )
            await conn.execute(
                "DELETE FROM translations t USING work_chapters wc "
                "WHERE t.work_chapter_id = wc.id AND wc.work_id=$1",
                work_id,
            )
            await conn.execute(
                "DELETE FROM chapters c USING work_chapters wc "
                "WHERE c.work_chapter_id = wc.id AND wc.work_id=$1",
                work_id,
            )
            await conn.execute(
                "DELETE FROM work_chapters WHERE work_id=$1", work_id,
            )
        log.info("done")
        return 0
    finally:
        await store.close()


async def _count(store: PostgresStore, work_id: int) -> dict[str, int]:
    rows = await store._pool.fetch(    # type: ignore[attr-defined]
        """
        SELECT 'work_chapters'   AS k, COUNT(*) AS n FROM work_chapters
          WHERE work_id=$1
        UNION ALL SELECT 'reading_history', COUNT(*) FROM reading_history rh
          JOIN work_chapters wc ON wc.id = rh.work_chapter_id WHERE wc.work_id=$1
        UNION ALL SELECT 'chapters', COUNT(*) FROM chapters c
          JOIN work_chapters wc ON wc.id = c.work_chapter_id WHERE wc.work_id=$1
        UNION ALL SELECT 'translations', COUNT(*) FROM translations t
          JOIN work_chapters wc ON wc.id = t.work_chapter_id WHERE wc.work_id=$1
        """,
        work_id,
    )
    return {r["k"]: int(r["n"]) for r in rows}


if __name__ == "__main__":
    argv = sys.argv[1:]
    assume_yes = "-y" in argv
    argv = [a for a in argv if a != "-y"]
    if len(argv) != 1:
        print("usage: python -m scripts.wipe_work_chapters <work_id> [-y]",
              file=sys.stderr)
        sys.exit(2)
    try:
        wid = int(argv[0])
    except ValueError:
        print(f"invalid work_id: {argv[0]!r}", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(wid, assume_yes=assume_yes)))
