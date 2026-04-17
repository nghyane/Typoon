"""Cloudflare cookie solver — shared by all CF-protected connectors.

Launches Edge with CDP, opens the target domain, waits for cf_clearance cookie.
Saves to CookieStore on success.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CDP_PORT = 9222
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
)
_EDGE_PATH = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"


async def solve(domain: str, timeout: int = 120) -> str:
    """Solve Cloudflare challenge for domain. Returns validated cookie string.

    Launches Edge browser if needed, opens domain, waits for cf_clearance,
    validates cookies work before returning.
    """
    import websockets

    url = f"https://{domain}"

    if not _cdp_alive():
        logger.info("Launching Edge with CDP...")
        _launch_edge()

    ws_url = _open_tab(url)
    logger.info("Got websocket: %s", ws_url[:60])

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as ws:
                await ws.send(json.dumps({
                    "id": 1,
                    "method": "Network.getCookies",
                    "params": {"urls": [url]},
                }))
                resp = json.loads(await ws.recv())
                cookies = resp.get("result", {}).get("cookies", [])
                if any(c["name"] == "cf_clearance" for c in cookies):
                    return "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        except Exception:
            pass
        await asyncio.sleep(2)

    raise TimeoutError(f"CF challenge not solved within {timeout}s for {domain}")


def _validate(domain: str, cookies: str) -> bool:
    """Check cookies work via httpx (same TLS fingerprint as actual requests)."""
    try:
        import httpx
        r = httpx.head(
            f"https://{domain}/home",
            headers={"Cookie": cookies, "User-Agent": _UA},
            follow_redirects=True, timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def _cdp_alive() -> bool:
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://localhost:{_CDP_PORT}/json/version", timeout=3)
        return b"Browser" in r.read()
    except Exception:
        return False


def _launch_edge() -> None:
    subprocess.run(
        ["osascript", "-e", 'tell application "Microsoft Edge" to quit'],
        capture_output=True, timeout=5,
    )
    time.sleep(2)
    subprocess.Popen(
        [_EDGE_PATH, f"--remote-debugging-port={_CDP_PORT}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if _cdp_alive():
            return
        time.sleep(0.5)
    raise RuntimeError("Failed to launch Edge with CDP")


def _open_tab(url: str) -> str:
    import urllib.request
    domain = url.split("//")[1].split("/")[0]

    # Check if tab already open
    tabs = json.loads(urllib.request.urlopen(f"http://localhost:{_CDP_PORT}/json").read())
    for t in tabs:
        if domain in t.get("url", ""):
            return t["webSocketDebuggerUrl"]

    # Open URL in Edge (works regardless of CDP version)
    subprocess.run(["open", "-a", "Microsoft Edge", url], timeout=5)
    time.sleep(2)

    # Find the new tab
    for _ in range(10):
        tabs = json.loads(urllib.request.urlopen(f"http://localhost:{_CDP_PORT}/json").read())
        for t in tabs:
            if domain in t.get("url", ""):
                return t["webSocketDebuggerUrl"]
        time.sleep(1)

    raise RuntimeError(f"Could not find tab for {domain} in Edge CDP")
