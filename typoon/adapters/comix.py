"""comix.to connector — discover chapters + extract image URLs.

Uses BrowserClient (CDP fetch) for CF-protected API calls.
Images on CDN are downloaded directly via httpx (no CF).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from ..auth.browser import BrowserClient
from ..auth.cookies import CookieStore
from ..types import ChapterVariant, DiscoveredChapter, SourceInfo

logger = logging.getLogger(__name__)

_BASE = "https://comix.to"
_DOMAIN = "comix.to"


class ComixConnector:
    """comix.to connector. Uses browser CDP for CF-protected requests."""

    def __init__(
        self,
        cookie_store: CookieStore | None = None,
        browser: BrowserClient | None = None,
    ) -> None:
        self._store = cookie_store or CookieStore()
        self._browser = browser or BrowserClient()

    @property
    def site_name(self) -> str:
        return _DOMAIN

    def accepts(self, url: str) -> bool:
        return _DOMAIN in url

    def is_authenticated(self) -> bool:
        """Check if browser has comix.to tab with CF clearance."""
        if not self._browser.is_running():
            return False
        if self._browser._find_tab(_DOMAIN) is None:
            return False
        # Probe: check cf_clearance cookie is still present
        cookies = self._store.get(_DOMAIN)
        return bool(cookies and "cf_clearance" in cookies)

    async def authenticate(self) -> None:
        """Launch Edge, open comix.to for CF challenge. User solves in browser."""
        from ..auth.cloudflare import solve
        self._browser.start()
        # Reload the page to trigger a fresh CF challenge
        tab_ws = self._browser._find_tab(_DOMAIN)
        if tab_ws:
            await self._reload_tab(tab_ws)
        cookies = await solve(_DOMAIN)
        self._store.put(_DOMAIN, cookies)

    @staticmethod
    async def _reload_tab(ws_url: str) -> None:
        """Reload an existing tab to trigger fresh CF challenge."""
        import websockets
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as ws:
                await ws.send(json.dumps({
                    "id": 1, "method": "Page.reload", "params": {"ignoreCache": True},
                }))
                await ws.recv()
        except Exception:
            pass  # tab may have navigated away, solve() will open fresh

    def http_headers(self) -> dict[str, str]:
        """Headers for direct image downloads (CDN, no CF)."""
        cookies = self._store.get(_DOMAIN) or ""
        return {
            "Cookie": cookies,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
        }

    async def _fetch(self, url: str) -> str:
        """Fetch via browser CDP. Auto re-auth on 403/failure."""
        try:
            return await self._browser.fetch(url, _DOMAIN)
        except Exception as e:
            logger.warning("Fetch failed (%s), re-authenticating…", e)
            await self.authenticate()
            return await self._browser.fetch(url, _DOMAIN)

    async def _fetch_json(self, url: str) -> dict:
        """Fetch JSON via browser CDP (bypasses CF)."""
        return json.loads(await self._fetch(url))

    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML via browser CDP (bypasses CF)."""
        return await self._fetch(url)

    @staticmethod
    def _parse_slug(url: str) -> str:
        m = re.search(r"/title/([^/?\s]+)", url)
        return m.group(1) if m else url.strip("/").split("/")[-1]

    async def discover(self, url: str) -> SourceInfo:
        slug = self._parse_slug(url)
        title = slug.split("-", 1)[1].replace("-", " ").title() if "-" in slug else slug
        prefix = slug.split("-")[0]

        all_items: list[dict] = []
        page = 1
        while True:
            data = await self._fetch_json(
                f"{_BASE}/api/v2/manga/{prefix}/chapters?limit=100&page={page}&order[number]=desc"
            )
            items = data.get("result", {}).get("items", [])
            if not items:
                break
            all_items.extend(items)
            page += 1

        by_number: dict[float, list[dict]] = defaultdict(list)
        for item in all_items:
            by_number[item["number"]].append(item)

        chapters = []
        for num in sorted(by_number):
            variants = [
                ChapterVariant(
                    id=str(item["chapter_id"]),
                    url=f"{_BASE}/title/{slug}/{item['chapter_id']}-chapter-{num}",
                    group=(item.get("scanlation_group") or {}).get("name"),
                    votes=item.get("votes", 0),
                )
                for item in by_number[num]
            ]
            chapters.append(DiscoveredChapter(number=num, variants=variants))

        # Detect source language from most common chapter language
        lang_counts: dict[str, int] = {}
        for item in all_items:
            lang = item.get("language", "en")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        detected_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"

        logger.info("comix.to: %s — %d chapters (%s)", slug, len(chapters), detected_lang)
        return SourceInfo(suggested_title=title, suggested_lang=detected_lang, chapters=chapters)

    async def get_page_urls(
        self, chapter: DiscoveredChapter, variant: ChapterVariant | None = None,
    ) -> list[str]:
        v = variant or chapter.best_variant
        html = await self._fetch_html(v.url)
        urls = list(dict.fromkeys(
            re.findall(r'https://[^"]+wowpic[^"]+\.webp', html)
        ))
        if not urls:
            raise RuntimeError(f"No images found for chapter {chapter.number}")
        return urls
