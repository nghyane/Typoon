#!/usr/bin/env python3
"""
Download a comix.to series for ComicScan testing.

Usage:
  python3 scripts/comix_download.py z0yj-ctrlaltresign
  python3 scripts/comix_download.py z0yj-ctrlaltresign -o tests/fixtures/ctrlaltresign
  python3 scripts/comix_download.py z0yj-ctrlaltresign -c 15

Requires: pip install httpx websockets

Cookie lifecycle:
  1. Check cached cookies in cache/comix_cookies.txt
  2. If invalid → launch Edge with CDP → wait for manual CF solve → extract cookies
  3. Cache cookies for reuse

Pipeline (no HEAD probes, no browser during download):
  1. /api/v2/manga/{slug}/chapters  → chapter list
  2. GET chapter HTML               → all image URLs (SSR-embedded)
  3. GET images                     → save to disk
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    print(msg, flush=True)

COOKIE_FILE = Path(__file__).resolve().parent.parent / "cache" / "comix_cookies.txt"
BASE = "https://comix.to"
CDP_PORT = 9222
EDGE_PATH = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
)


# ── Cookie management ───────────────────────────────────────────────


def load_cookies() -> str | None:
    if COOKIE_FILE.exists():
        c = COOKIE_FILE.read_text().strip()
        return c or None
    return None


def save_cookies(cookies: str):
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(cookies)


def cookies_valid(cookies: str) -> bool:
    try:
        r = httpx.head(f"{BASE}/home", headers={"Cookie": cookies, "User-Agent": UA},
                       follow_redirects=True, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def cdp_alive() -> bool:
    try:
        return "Browser" in httpx.get(f"http://localhost:{CDP_PORT}/json/version", timeout=3).text
    except Exception:
        return False


def launch_edge():
    subprocess.run(["osascript", "-e", 'tell application "Microsoft Edge" to quit'],
                   capture_output=True, timeout=5)
    time.sleep(2)
    subprocess.Popen([EDGE_PATH, f"--remote-debugging-port={CDP_PORT}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if cdp_alive():
            return True
        time.sleep(0.5)
    return False


async def grab_cookies_cdp(timeout: int = 120) -> str:
    import websockets
    tabs = httpx.get(f"http://localhost:{CDP_PORT}/json").json()
    ws_url = next((t["webSocketDebuggerUrl"] for t in tabs if "comix.to" in t.get("url", "")), None)
    if not ws_url:
        ws_url = httpx.get(f"http://localhost:{CDP_PORT}/json/new?{BASE}/home").json()["webSocketDebuggerUrl"]

    log("⏳ Solve CF challenge in Edge...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Network.getCookies", "params": {"urls": [BASE]}}))
                resp = json.loads(await ws.recv())
                cookies = resp.get("result", {}).get("cookies", [])
                if any(c["name"] == "cf_clearance" for c in cookies):
                    return "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("CF challenge not solved within timeout")


async def ensure_cookies() -> str:
    saved = load_cookies()
    if saved and cookies_valid(saved):
        log("✓ Cookies valid")
        return saved

    log("✗ Cookies expired or missing")
    if not cdp_alive():
        log("Launching Edge with CDP...")
        if not launch_edge():
            sys.exit("Failed to launch Edge")
    log(f"✓ Edge CDP on port {CDP_PORT}")

    cookies = await grab_cookies_cdp()
    if not cookies_valid(cookies):
        sys.exit("Extracted cookies invalid")
    save_cookies(cookies)
    log(f"✓ Cookies saved → {COOKIE_FILE}")
    return cookies


# ── Step 1: Chapter list via API ────────────────────────────────────


async def fetch_chapter_list(client: httpx.AsyncClient, slug: str) -> list[tuple[float, int]]:
    """Return [(chapter_number, chapter_id)] via /api/v2/manga/{prefix}/chapters."""
    prefix = slug.split("-")[0]
    all_items: list[dict] = []
    page = 1
    while True:
        resp = await client.get(f"{BASE}/api/v2/manga/{prefix}/chapters",
                                params={"limit": 100, "page": page, "order[number]": "desc"})
        items = resp.json().get("result", {}).get("items", [])
        if not items:
            break
        all_items.extend(items)
        page += 1

    by_number: dict[float, list[dict]] = defaultdict(list)
    for item in all_items:
        by_number[item["number"]].append(item)

    result: list[tuple[float, int]] = []
    group_stats: dict[str, int] = defaultdict(int)
    for ch in sorted(by_number):
        picked = max(by_number[ch], key=lambda e: e.get("votes", 0))
        gname = (picked.get("scanlation_group") or {}).get("name", "?")
        group_stats[gname] += 1
        result.append((ch, picked["chapter_id"]))

    groups_summary = ", ".join(f"{g} ×{c}" for g, c in sorted(group_stats.items(), key=lambda x: -x[1]))
    log(f"  Sources: {groups_summary}")
    return result


# ── Step 2+3: Fetch chapter HTML → extract image URLs → download ───


def _ch_label(ch_num: float) -> str:
    """Format chapter number for dir names and logs: 3 -> '003', 3.5 -> '003.5'."""
    if ch_num == int(ch_num):
        return f"{int(ch_num):03d}"
    return f"{int(ch_num):03d}.{str(ch_num).split('.')[1]}"


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.pages = 0
        self.t0 = time.time()
        self._lock = asyncio.Lock()

    async def finish_chapter(self, ch_num: float, page_count: int, skipped: bool = False):
        async with self._lock:
            self.done += 1
            self.pages += page_count
            elapsed = time.time() - self.t0
            status = "skip" if skipped else f"{page_count}p"
            log(f"  [{self.done}/{self.total}] Ch.{_ch_label(ch_num)} {status}  ({elapsed:.0f}s)")


async def download_chapter(
    client: httpx.AsyncClient, slug: str,
    ch_num: float, ch_id: int, output_dir: Path,
    sem: asyncio.Semaphore, progress: Progress,
    trim: int = 0,
) -> int:
    async with sem:
        ch_dir = output_dir / f"ch{_ch_label(ch_num)}"

        # Skip if directory already has images
        existing = list(ch_dir.glob("*.webp")) if ch_dir.exists() else []
        if existing and all(f.stat().st_size > 0 for f in existing):
            await progress.finish_chapter(ch_num, len(existing), skipped=True)
            return len(existing)

        ch_dir.mkdir(parents=True, exist_ok=True)

        # Fetch chapter HTML (contains all image URLs in SSR)
        for attempt in range(3):
            try:
                resp = await client.get(f"{BASE}/title/{slug}/{ch_id}-chapter-{ch_num}")
                break
            except Exception:
                if attempt == 2:
                    await progress.finish_chapter(ch_num, 0)
                    return 0
                await asyncio.sleep(1 * (attempt + 1))
        urls = list(dict.fromkeys(re.findall(r'https://[^"]+wowpic[^"]+\.webp', resp.text)))

        # Trim cover/credits pages from start and end
        if trim > 0 and len(urls) > trim * 2:
            urls = urls[trim:-trim]

        if not urls:
            await progress.finish_chapter(ch_num, 0)
            return 0

        # Download missing images
        tasks = []
        for i, url in enumerate(urls, 1):
            fp = ch_dir / f"{i:02d}.webp"
            if fp.exists() and fp.stat().st_size > 0:
                continue
            tasks.append(_save(client, url, fp))

        if tasks:
            await asyncio.gather(*tasks)
            await progress.finish_chapter(ch_num, len(urls))
        else:
            await progress.finish_chapter(ch_num, len(urls), skipped=True)
        return len(urls)


async def _save(client: httpx.AsyncClient, url: str, path: Path):
    for attempt in range(3):
        try:
            resp = await client.get(url, timeout=30)
            if resp.status_code == 200:
                path.write_bytes(resp.content)
                return
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1 * (attempt + 1))


# ── Main ────────────────────────────────────────────────────────────


async def main():
    p = argparse.ArgumentParser(description="Download comix.to series")
    p.add_argument("series", help="Series slug, e.g. z0yj-ctrlaltresign")
    p.add_argument("-o", "--output", help="Output directory")
    p.add_argument("-c", "--concurrency", type=int, default=10)
    p.add_argument("-t", "--trim", type=int, default=0,
                   help="Trim N images from start and end of each chapter (remove cover/credits)")
    args = p.parse_args()

    output = Path(args.output) if args.output else Path(f"tests/fixtures/{args.series}")
    cookies = await ensure_cookies()

    t0 = time.time()
    limits = httpx.Limits(max_connections=30, max_keepalive_connections=15)
    async with httpx.AsyncClient(
        headers={"Cookie": cookies, "User-Agent": UA},
        follow_redirects=True, timeout=30, limits=limits,
    ) as client:
        log(f"\n=== Fetching chapter list ===")
        chapters = await fetch_chapter_list(client, args.series)
        log(f"  {len(chapters)} chapters (Ch.{chapters[0][0]}–{chapters[-1][0]})")

        log(f"\n=== Downloading ({args.concurrency} parallel) ===")
        output.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(args.concurrency)
        progress = Progress(len(chapters))

        async def with_timeout(ch, cid):
            try:
                return await asyncio.wait_for(
                    download_chapter(client, args.series, ch, cid, output, sem, progress, trim=args.trim),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                await progress.finish_chapter(ch, 0)
                log(f"    ⚠ Ch.{_ch_label(ch)} timed out")
                return 0

        results = await asyncio.gather(*(
            with_timeout(ch, cid) for ch, cid in chapters
        ))

    total_pages = sum(results)
    size_mb = sum(f.stat().st_size for f in output.rglob("*.webp")) / 1024 / 1024
    log(f"\n=== Done in {time.time() - t0:.1f}s ===")
    log(f"  {len(chapters)} chapters · {total_pages} pages · {size_mb:.0f} MB → {output}")


if __name__ == "__main__":
    asyncio.run(main())
