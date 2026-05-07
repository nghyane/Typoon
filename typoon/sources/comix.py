"""comix.to connector — discover chapters + extract image URLs.

Uses BrowserClient (CDP fetch) for CF-protected API calls.
Token discovery: captures anti-CSRF token from browser's own API calls.
Images on CDN are downloaded directly via httpx (no CF).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlparse

from ..auth.browser import BrowserClient
from ..auth.cookies import CookieStore
from ..domain.project import ChapterVariant, DiscoveredChapter, SourceInfo

logger = logging.getLogger(__name__)

_BASE = "https://comix.to"
_DOMAIN = "comix.to"
_SOURCE_LANG = "en"  # comix.to serves English scanlations


class ComixConnector:
    """comix.to connector. Captures API token from browser then queries API."""

    def __init__(
        self,
        cookie_store: CookieStore | None = None,
        browser: BrowserClient | None = None,
    ) -> None:
        self._store = cookie_store or CookieStore()
        self._browser = browser or BrowserClient()

    @property
    def site_id(self) -> str:
        """Stable identifier — used by the API source listing."""
        return _DOMAIN

    @property
    def site_name(self) -> str:
        return "Comix"

    @property
    def source_lang(self) -> str:
        return _SOURCE_LANG

    @property
    def example_url(self) -> str:
        return f"{_BASE}/title/<id>-<slug>"

    @property
    def description(self) -> str:
        return "English scanlation aggregator. Cloudflare-protected; works in dev."

    def accepts(self, url: str) -> bool:
        return _DOMAIN in url

    def is_authenticated(self) -> bool:
        """Check if browser has comix.to tab with CF clearance."""
        if not self._browser.is_running():
            return False
        if self._browser._find_tab(_DOMAIN) is None:
            return False
        cookies = self._store.get(_DOMAIN)
        return bool(cookies and "cf_clearance" in cookies)

    async def authenticate(self) -> None:
        """Launch Edge, open comix.to for CF challenge. User solves in browser."""
        from ..auth.cloudflare import solve
        self._browser.start()
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
            pass

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
        slug   = self._parse_slug(url)
        prefix = slug.split("-")[0]

        detail = await self._fetch_detail(prefix)
        if detail is None:
            raise RuntimeError(f"comix.to: cannot fetch manga detail for {prefix}")

        token = await self._capture_token(url)
        if token:
            chapters = await self._discover_via_api(prefix, slug, token)
        else:
            logger.warning("Token capture failed, falling back to DOM scraping")
            chapters = await self._scrape_from_dom(url, slug)

        logger.info("comix.to: %s — %d chapters", slug, len(chapters))
        return SourceInfo(
            suggested_title=detail["title"],
            cover_url=(detail.get("poster") or {}).get("large"),
            description=detail.get("synopsis"),
            chapters=chapters,
        )

    async def _fetch_detail(self, prefix: str) -> dict | None:
        """Fetch manga detail. Public endpoint — no CF, no token needed."""
        import httpx
        url = f"{_BASE}/api/v2/manga/{prefix}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Manga detail fetch failed: %s", e)
            return None
        m = data.get("result") if isinstance(data, dict) else None
        if not isinstance(m, dict) or not m.get("title"):
            return None
        return m

    async def _capture_token(self, url: str) -> str | None:
        """Capture anti-CSRF token from browser's API request.

        Navigates to the title page using Page.navigate (single WebSocket
        session), captures the API call including the &_= token.
        """
        try:
            captured_url = await self._browser.capture_network_request(
                _DOMAIN,
                "/api/v2/manga/",
                navigate_url=url,
                reload=False,
                timeout=30,
            )
            if not captured_url:
                return None

            # Extract token from captured URL
            parsed = urlparse(captured_url)
            params = dict(parse_qsl(parsed.query))
            return params.get("_")
        except Exception as e:
            logger.warning("Token capture failed: %s", e)
            return None

    async def _discover_via_api(self, prefix: str, slug: str, token: str) -> list[DiscoveredChapter]:
        """Discover chapters via API using captured token."""
        all_items: list[dict] = []
        page = 1
        while True:
            try:
                url = (
                    f"{_BASE}/api/v2/manga/{prefix}/chapters"
                    f"?limit=100&page={page}&order[number]=desc"
                    f"&time=1&_={token}"
                )
                data = await self._fetch_json(url)
                items = data.get("result", {}).get("items", [])
                if not items:
                    break
                all_items.extend(items)
                page += 1
            except Exception as e:
                logger.warning("API page %d failed: %s", page, e)
                break

        by_number: dict[float, list[dict]] = defaultdict(list)
        for item in all_items:
            by_number[float(item["number"])].append(item)

        chapters = []
        for num in sorted(by_number):
            items = by_number[num]
            chapter_title = next(
                (it["name"] for it in items if it.get("name")),
                None,
            )
            variants = [
                ChapterVariant(
                    id=str(item["chapter_id"]),
                    url=f"{_BASE}/title/{slug}/{item['chapter_id']}-chapter-{num}",
                    group=(item.get("scanlation_group") or {}).get("name"),
                    votes=item.get("votes", 0),
                )
                for item in items
            ]
            chapters.append(DiscoveredChapter(number=num, title=chapter_title, variants=variants))

        return chapters

    async def _scrape_from_dom(self, url: str, slug: str) -> list[DiscoveredChapter]:
        """Fallback: scrape chapter links from rendered DOM via browser CDP."""
        import asyncio

        all_chapters: dict[float, list[str]] = defaultdict(list)

        # Navigate to title page
        await self._browser.execute_js(
            f'window.location.href = "{url}"',
            _DOMAIN, timeout=30,
        )
        await asyncio.sleep(3)

        page_target = 1
        while True:
            # Scrape chapter links from current DOM
            scrape_js = '''
            (() => {
                const links = [...document.querySelectorAll('a[href*="chapter"]')];
                return links.map(a => {
                    const match = a.href.match(/-chapter-([\\d.]+)$/);
                    return {
                        num: match ? match[1] : null,
                        href: a.href,
                    };
                }).filter(l => l.num !== null);
            })()
            '''
            result = await self._browser.execute_js(scrape_js, _DOMAIN, timeout=15)
            dom_chapters = result.get("value", [])

            new_count = 0
            for ch in dom_chapters:
                try:
                    num = float(ch["num"])
                except (ValueError, KeyError):
                    continue
                if ch["href"] not in all_chapters[num]:
                    all_chapters[num].append(ch["href"])
                    new_count += 1

            # Try to click Next page
            click_js = f'''
            (() => {{
                const next = document.querySelector('.page-link[href="#{page_target + 1}"]');
                if (next) {{
                    next.click();
                    return 'clicked';
                }}
                const nextBtn = [...document.querySelectorAll('.page-link')].find(a => a.textContent.trim() === 'Next');
                if (nextBtn && !nextBtn.closest('.disabled')) {{
                    nextBtn.click();
                    return 'clicked';
                }}
                return 'done';
            }})()
            '''
            result = await self._browser.execute_js(click_js, _DOMAIN, timeout=15)
            click_result = result.get("value", "")

            if click_result == "done" or (new_count == 0 and page_target > 1):
                break

            await asyncio.sleep(3)
            page_target += 1
            if page_target > 50:
                break

        chapters = []
        for num in sorted(all_chapters):
            variants = [
                ChapterVariant(id=f"ch{num}_{i}", url=href)
                for i, href in enumerate(all_chapters[num])
            ]
            chapters.append(DiscoveredChapter(number=num, variants=variants))

        return chapters

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
