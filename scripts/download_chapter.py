"""Download one chapter from MangaDex by URL → save pages as JPEG.

Usage:
  python3 scripts/download_chapter.py CHAPTER_URL OUT_DIR

CHAPTER_URL is any MangaDex chapter URL, e.g.
  https://mangadex.org/chapter/abc-...-uuid

Pages written as 0000.jpg, 0001.jpg, ... into OUT_DIR.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import httpx

_UUID_RE = re.compile(r"/chapter/([0-9a-f-]{36})")


async def main(url: str, out_dir: Path) -> None:
    m = _UUID_RE.search(url)
    if not m:
        raise SystemExit(f"Cannot extract chapter UUID from URL: {url}")
    chapter_id = m.group(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(http2=True, timeout=60.0) as client:
        meta_url = f"https://api.mangadex.org/at-home/server/{chapter_id}"
        print(f"GET {meta_url}")
        r = await client.get(meta_url)
        r.raise_for_status()
        meta = r.json()
        base = meta["baseUrl"]
        h = meta["chapter"]["hash"]
        files: list[str] = meta["chapter"]["data"]
        print(f"chapter: {len(files)} pages, hash={h}")

        for i, fname in enumerate(files):
            ext = fname.rsplit(".", 1)[-1].lower()
            page_url = f"{base}/data/{h}/{fname}"
            out_path = out_dir / f"{i:04d}.{ext}"
            if out_path.exists():
                print(f"[{i:02d}] skip (exists) {out_path.name}")
                continue
            resp = await client.get(page_url)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            print(f"[{i:02d}] {out_path.name} ({len(resp.content) // 1024} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: download_chapter.py CHAPTER_URL OUT_DIR")
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
