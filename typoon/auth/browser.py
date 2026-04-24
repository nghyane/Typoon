"""Browser client — CDP fetch for CF-protected sites.

Uses Edge GUI with CDP. Fetches via JS fetch() in an existing tab.
Lightweight: 1 websocket message per request.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.request

logger = logging.getLogger(__name__)

_CDP_PORT = 9222
_EDGE_PATH = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"


class BrowserClient:
    """Fetch URLs through Edge browser via CDP. Bypasses CF TLS fingerprinting."""

    def __init__(self, port: int = _CDP_PORT) -> None:
        self._port = port
        self._ws_url: str | None = None

    def is_running(self) -> bool:
        try:
            r = urllib.request.urlopen(f"http://localhost:{self._port}/json/version", timeout=3)
            return b"Browser" in r.read()
        except Exception:
            return False

    def start(self) -> None:
        """Launch Edge with CDP if not running."""
        if self.is_running():
            return
        logger.info("Launching Edge with CDP...")
        subprocess.run(
            ["osascript", "-e", 'tell application "Microsoft Edge" to quit'],
            capture_output=True, timeout=5,
        )
        time.sleep(2)
        subprocess.Popen(
            [_EDGE_PATH, f"--remote-debugging-port={self._port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(20):
            if self.is_running():
                return
            time.sleep(0.5)
        raise RuntimeError("Failed to launch Edge with CDP")

    def _find_tab(self, domain: str) -> str | None:
        """Find websocket URL for a tab containing domain."""
        tabs = json.loads(
            urllib.request.urlopen(f"http://localhost:{self._port}/json").read()
        )
        for t in tabs:
            if domain in t.get("url", ""):
                return t["webSocketDebuggerUrl"]
        return None

    def _open_tab(self, url: str, domain: str) -> str:
        """Open URL in Edge, return websocket URL."""
        ws = self._find_tab(domain)
        if ws:
            return ws
        subprocess.run(["open", "-a", "Microsoft Edge", url], timeout=5)
        time.sleep(2)
        for _ in range(10):
            ws = self._find_tab(domain)
            if ws:
                return ws
            time.sleep(1)
        raise RuntimeError(f"Could not find tab for {domain}")

    async def fetch(self, url: str, domain: str, timeout: int = 30) -> str:
        """Fetch URL through browser. Returns response text."""
        import asyncio, websockets

        ws_url = self._find_tab(domain)
        if not ws_url:
            ws_url = self._open_tab(f"https://{domain}", domain)

        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            await ws.send(json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": f'fetch("{url}", {{credentials: "include"}}).then(r => r.ok ? r.text() : Promise.reject(r.status))', "awaitPromise": True},
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            result = resp.get("result", {}).get("result", {})
            if result.get("type") == "string":
                return result["value"]
            err = resp.get("result", {}).get("exceptionDetails", {})
            raise RuntimeError(f"Browser fetch failed: {err or result}")

    async def capture_network_request(self, domain: str, url_pattern: str, navigate_url: str | None = None, reload: bool = True, timeout: int = 20) -> str | None:
        """Capture first network request matching pattern. Returns full URL.

        If navigate_url is given, navigates to that URL first, capturing
        requests fired DURING page load (e.g. React API calls).
        If reload=True, reloads the current page to trigger fresh requests.
        """
        import asyncio, websockets

        ws_url = self._find_tab(domain)
        if not ws_url:
            ws_url = self._open_tab(f"https://{domain}", domain)

        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            # Enable network domain
            await ws.send(json.dumps({"id": 1, "method": "Network.enable"}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            if navigate_url:
                # Navigate to target URL, capture requests during load
                await ws.send(json.dumps({
                    "id": 2,
                    "method": "Page.navigate",
                    "params": {"url": navigate_url},
                }))

                # Drain messages: capture API calls + wait for navigate response
                navigate_done = False
                start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(msg)
                        # Check for matching API requests
                        if data.get("method") == "Network.requestWillBeSent":
                            url = data["params"]["request"]["url"]
                            if url_pattern in url:
                                return url
                        # Check for navigate completion
                        if data.get("id") == 2:
                            navigate_done = True
                    except asyncio.TimeoutError:
                        break

                # After navigate completes, wait a bit more for lazy-loaded API calls
                if navigate_done:
                    await asyncio.sleep(3)
                    deadline = asyncio.get_event_loop().time() + 10
                    while asyncio.get_event_loop().time() < deadline:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=3)
                            data = json.loads(msg)
                            if data.get("method") == "Network.requestWillBeSent":
                                url = data["params"]["request"]["url"]
                                if url_pattern in url:
                                    return url
                        except asyncio.TimeoutError:
                            break
                return None

            elif reload:
                # Reload page to trigger requests
                await ws.send(json.dumps({"id": 3, "method": "Page.reload", "params": {"ignoreCache": True}}))

            # Listen for matching requests
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    if data.get("method") == "Network.requestWillBeSent":
                        url = data["params"]["request"]["url"]
                        if url_pattern in url:
                            return url
                except asyncio.TimeoutError:
                    break
            return None

    async def execute_js(self, expression: str, domain: str, timeout: int = 30) -> dict:
        """Execute JS expression in browser tab, return value."""
        import asyncio, websockets

        ws_url = self._find_tab(domain)
        if not ws_url:
            ws_url = self._open_tab(f"https://{domain}", domain)

        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            await ws.send(json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "awaitPromise": True, "returnByValue": True},
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            result = resp.get("result", {}).get("result", {})
            return result
