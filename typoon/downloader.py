"""Shared image downloader — used by all connectors.

Downloads a list of URLs to a local folder with retry + skip existing.
Connector provides URLs + headers, downloader handles the rest.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


async def download_images(
    urls: list[str],
    dest: Path,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    concurrency: int = 5,
    skip_existing: bool = True,
) -> Path:
    """Download images to dest/. Returns dest path.

    Args:
        skip_existing: If True (default), skip files already on disk.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        headers=headers or {}, follow_redirects=True, timeout=30,
    ) as client:
        tasks = [
            _download_one(client, url, dest / f"{i + 1:03d}{_ext(url)}",
                          sem, max_retries, skip_existing)
            for i, url in enumerate(urls)
        ]
        await asyncio.gather(*tasks)

    downloaded = len(list(dest.iterdir()))
    logger.info("Downloaded %d/%d images → %s", downloaded, len(urls), dest)
    return dest


async def _download_one(
    client: httpx.AsyncClient, url: str, fp: Path,
    sem: asyncio.Semaphore, max_retries: int, skip_existing: bool,
) -> None:
    if skip_existing and fp.exists() and fp.stat().st_size > 0:
        return
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    fp.write_bytes(resp.content)
                    return
            except Exception:
                pass
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
        logger.warning("Failed after %d attempts: %s", max_retries, url)


def _ext(url: str) -> str:
    path = url.split("?")[0]
    for ext in (".webp", ".jpg", ".jpeg", ".png"):
        if path.endswith(ext):
            return ext
    return ".webp"
