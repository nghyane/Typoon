"""E2E debug: translate chapter 6 with artifact sink.

Usage:
    python scripts/e2e_translate.py [chapter_id]

Writes everything to `debug-runs/e2e_c<chapter_id>/`. Designed to be
runnable standalone — no worker, no draft state changes; pure read-only
verification of the translate pipeline against real DB + storage.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from typoon.adapters.chapter_archive import prepared_key
from typoon.adapters.ctx import make_ctx
from typoon.adapters.loader import load_scanned, open_prepared_reader
from typoon.adapters.storage_registry import build_storage
from typoon.config import load_config
from typoon.runs.artifacts import FileArtifactSink
from typoon.runs.events import LoggingHook
from typoon.stages.translate import translate_chapter
from typoon.storage import PostgresStore


async def main(chapter_id: int) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config, paths = load_config()
    db = await PostgresStore.open(config.database_url)
    stores = build_storage(config, paths)

    try:
        info = await db.get_chapter(chapter_id)
        if info is None:
            print(f"chapter {chapter_id} not found", file=sys.stderr)
            return 2
        print(f"chapter {chapter_id}: source_lang={info['source_lang']} pages={info['page_count']}")

        # Pick the most recent draft so we use its (draft_id, target_lang).
        # If none, default to (draft_id=-1, target_lang=vi) so we exercise
        # translate without touching DB state.
        draft = await _latest_draft(db, chapter_id)
        if draft is None:
            print("no draft; using ephemeral (draft_id=-1, target_lang=vi)")
            draft_id = -1
            target_lang = "vi"
            material_id = 0
            owner_id = 0
        else:
            draft_id = draft["id"]
            target_lang = draft["target_lang"]
            material_id, owner_id = await _draft_owner(db, chapter_id)
            print(f"using draft {draft_id} target_lang={target_lang}")

        sink = FileArtifactSink(
            Path("debug-runs"),
            f"e2e_c{chapter_id}",
            clean=True,
        )
        print(f"artifacts → {sink.root}")

        ctx = make_ctx(
            chapter_id=chapter_id,
            draft_id=draft_id,
            chapter_position=info.get("position", 0),
            material_id=material_id,
            owner_id=owner_id,
            source_lang=info["source_lang"],
            target_lang=target_lang,
            store=db,
            config=config,
            hook=LoggingHook(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with await open_prepared_reader(
                stores.pipeline, prepared_key(chapter_id), Path(tmp),
            ) as reader:
                scanned = await load_scanned(reader, db, chapter_id)
                print(f"scanned: {sum(len(p.bubbles) for p in scanned.pages)} bubbles across {len(scanned.pages)} pages")
                translated, brief = await translate_chapter(
                    scanned, reader, ctx,
                    artifacts=sink,
                )

        ops_count = sum(1 for p in translated.pages for b in p.bubbles)
        skip = sum(1 for p in translated.pages for b in p.bubbles if b.kind == "skip")
        translated_n = sum(1 for p in translated.pages for b in p.bubbles if b.translated_text)
        print(
            f"\ndone: {ops_count} bubbles  "
            f"translated={translated_n}  "
            f"skip={skip}  "
            f"noise_keys={len(brief.noise_keys)}  "
            f"noise_pages={len(brief.noise_pages)}",
        )
        return 0
    finally:
        await stores.aclose()
        await db.close()


async def _latest_draft(db, chapter_id: int) -> dict | None:
    rows = await db._pool.fetch(  # type: ignore[attr-defined]
        "SELECT id, target_lang FROM translation_drafts "
        "WHERE chapter_id=$1 ORDER BY updated_at DESC LIMIT 1",
        chapter_id,
    )
    return dict(rows[0]) if rows else None


async def _draft_owner(db, chapter_id: int) -> tuple[int, int]:
    row = await db._pool.fetchrow(  # type: ignore[attr-defined]
        "SELECT m.id AS material_id, td.created_by AS owner_id "
        "FROM translation_drafts td "
        "JOIN chapters c ON c.id = td.chapter_id "
        "JOIN work_chapters wc ON wc.id = c.work_chapter_id "
        "JOIN materials m ON m.id = wc.work_id "
        "WHERE td.chapter_id=$1 "
        "ORDER BY td.updated_at DESC LIMIT 1",
        chapter_id,
    )
    if row is None:
        return 0, 0
    return row["material_id"], row["owner_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_id", type=int, nargs="?", default=6)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.chapter_id)))
