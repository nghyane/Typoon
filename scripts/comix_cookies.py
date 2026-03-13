#!/usr/bin/env python3
"""
Auto-grab Cloudflare cookies from Edge via CDP.

Usage:
  python3 scripts/comix_cookies.py          # grab & save cookies
  python3 scripts/comix_cookies.py --check  # check if saved cookies still work

Flow:
  1. Check if saved cookies are still valid (curl test)
  2. If expired → launch Edge with CDP, open comix.to, wait for CF solve
  3. Extract cookies via CDP websocket, save to cache file
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

COOKIE_FILE = Path(__file__).parent.parent / "cache" / "comix_cookies.txt"
DOMAIN = "https://comix.to"
TEST_URL = f"{DOMAIN}/home"
CDP_PORT = 9222
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
)
EDGE_PATH = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"


def load_cookies() -> str | None:
    if COOKIE_FILE.exists():
        cookies = COOKIE_FILE.read_text().strip()
        if cookies:
            return cookies
    return None


def test_cookies(cookies: str) -> bool:
    """Return True if cookies pass CF check."""
    try:
        r = subprocess.run(
            ["curl", "-sI", TEST_URL, "-H", f"Cookie: {cookies}", "-H", f"User-Agent: {UA}",
             "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10"],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip() == "200"
    except Exception:
        return False


def is_edge_cdp_up() -> bool:
    try:
        r = subprocess.run(
            ["curl", "-s", f"http://localhost:{CDP_PORT}/json/version"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and "Browser" in r.stdout
    except Exception:
        return False


def launch_edge_with_cdp():
    """Quit Edge if running, relaunch with CDP."""
    subprocess.run(
        ["osascript", "-e", 'tell application "Microsoft Edge" to quit'],
        capture_output=True, timeout=5,
    )
    time.sleep(2)
    subprocess.Popen(
        [EDGE_PATH, f"--remote-debugging-port={CDP_PORT}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for CDP
    for _ in range(20):
        if is_edge_cdp_up():
            return True
        time.sleep(0.5)
    return False


def open_tab(url: str):
    """Open a URL in Edge via CDP."""
    import urllib.request
    tabs = json.loads(urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json").read())
    for t in tabs:
        if DOMAIN in t.get("url", ""):
            # Navigate existing tab
            ws_id = t["id"]
            urllib.request.urlopen(
                f"http://localhost:{CDP_PORT}/json/activate/{ws_id}"
            )
            return t["webSocketDebuggerUrl"]
    # Create new tab
    resp = json.loads(
        urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json/new?{url}").read()
    )
    return resp["webSocketDebuggerUrl"]


async def extract_cookies_cdp(ws_url: str) -> str:
    import websockets
    async with websockets.connect(ws_url, max_size=10_000_000) as ws:
        await ws.send(json.dumps({
            "id": 1,
            "method": "Network.getCookies",
            "params": {"urls": [DOMAIN]},
        }))
        resp = json.loads(await ws.recv())
        cookies = resp.get("result", {}).get("cookies", [])
        return "; ".join(f"{c['name']}={c['value']}" for c in cookies)


async def wait_for_cf_clearance(ws_url: str, timeout: int = 120) -> str:
    """Poll until cf_clearance cookie appears."""
    import websockets
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as ws:
                await ws.send(json.dumps({
                    "id": 1,
                    "method": "Network.getCookies",
                    "params": {"urls": [DOMAIN]},
                }))
                resp = json.loads(await ws.recv())
                cookies = resp.get("result", {}).get("cookies", [])
                names = {c["name"] for c in cookies}
                if "cf_clearance" in names:
                    cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
                    return cookie_str
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("CF challenge not solved within timeout")


def save_cookies(cookies: str):
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(cookies)


def main():
    check_only = "--check" in sys.argv

    # 1. Try saved cookies
    saved = load_cookies()
    if saved:
        if test_cookies(saved):
            print(f"✓ Cookies valid ({COOKIE_FILE})")
            if not check_only:
                print(saved)
            return 0
        else:
            print("✗ Saved cookies expired")

    if check_only:
        return 1

    # 2. Launch Edge with CDP
    print("Launching Edge with CDP...")
    if not is_edge_cdp_up():
        if not launch_edge_with_cdp():
            print("✗ Failed to launch Edge with CDP")
            return 1
    print(f"✓ Edge CDP on port {CDP_PORT}")

    # 3. Open comix.to
    print(f"Opening {DOMAIN}...")
    ws_url = open_tab(f"{DOMAIN}/home")
    print("⏳ Waiting for Cloudflare challenge (solve it in Edge)...")

    # 4. Wait for cf_clearance
    try:
        cookies = asyncio.run(wait_for_cf_clearance(ws_url, timeout=120))
    except TimeoutError:
        print("✗ Timeout waiting for CF challenge")
        return 1

    # 5. Validate & save
    if test_cookies(cookies):
        save_cookies(cookies)
        print(f"✓ Cookies saved to {COOKIE_FILE}")
        print(cookies)
        return 0
    else:
        print("✗ Extracted cookies don't work")
        return 1


if __name__ == "__main__":
    sys.exit(main())
