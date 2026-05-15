"""Run per-chapter audit in parallel across all chapters with done drafts.

One subprocess per chapter (true OS parallelism). Each subprocess writes
to its own `debug-runs/audit/<chapter_id>/` directory so they never
collide on disk. Stdout lines from children are tagged with the chapter
id and streamed live so progress is visible.

Usage:
    python scripts/audit_all.py [--sample N] [--concurrent-chapters N]
                                [--concurrent-per-chapter N]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def _list_chapters() -> list[int]:
    import asyncpg
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        rows = await conn.fetch("""
            SELECT DISTINCT c.id
            FROM translation_drafts td
            JOIN chapters c ON c.id = td.chapter_id
            WHERE td.state = 'done'
              AND c.source_lang IS NOT NULL
            ORDER BY c.id
        """)
    finally:
        await conn.close()
    return [r["id"] for r in rows]


async def _run_one(
    chapter_id: int,
    sample: int | None,
    per_chapter: int,
    sem: asyncio.Semaphore,
) -> tuple[int, int, float]:
    """Run audit_chapter.py for one chapter. Returns (chapter_id, rc, secs)."""
    async with sem:
        t0 = time.monotonic()
        cmd = [
            sys.executable, "-u", "scripts/audit_chapter.py", str(chapter_id),
            "--concurrent", str(per_chapter),
        ]
        if sample is not None:
            cmd += ["--sample", str(sample)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def reader() -> None:
            assert proc.stdout is not None
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[c{chapter_id}] {text}", flush=True)

        read_task = asyncio.create_task(reader())
        rc = await proc.wait()
        await read_task

        return chapter_id, rc, time.monotonic() - t0


async def main(
    sample: int | None,
    concurrent_chapters: int,
    concurrent_per_chapter: int,
) -> None:
    chapters = await _list_chapters()
    print(
        f"[audit_all] {len(chapters)} chapters | "
        f"chapters in parallel: {concurrent_chapters} | "
        f"judge calls per chapter: {concurrent_per_chapter} | "
        f"sample/chapter: {sample or 'all'}",
        flush=True,
    )

    sem = asyncio.Semaphore(concurrent_chapters)
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _run_one(cid, sample, concurrent_per_chapter, sem) for cid in chapters
    ])
    elapsed = time.monotonic() - t0

    print()
    print("=" * 60)
    print(f"FINISHED in {elapsed:.0f}s")
    print("=" * 60)
    failed = [(cid, rc, secs) for cid, rc, secs in results if rc != 0]
    for cid, rc, secs in results:
        flag = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  c{cid:>4}  {flag:10s}  {secs:.0f}s")
    if failed:
        print(f"\n{len(failed)} chapters failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--concurrent-chapters", type=int, default=6)
    parser.add_argument("--concurrent-per-chapter", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main(
        args.sample,
        args.concurrent_chapters,
        args.concurrent_per_chapter,
    ))
