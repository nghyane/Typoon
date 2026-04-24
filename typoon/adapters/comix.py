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
from ..domain.project import ChapterVariant, DiscoveredChapter, SourceInfo

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

        # First try: scrape chapter links from rendered DOM via browser
        chapters = await self._scrape_from_dom(url, slug)

        # Fallback: try API (needs anti-CSRF token, often fails)
        if not chapters:
            logger.info("DOM scraping empty, trying API fallback...")
            chapters = await self._discover_via_api(prefix, slug)

        # Detect language from title patterns
        detected_lang = "en"
        if "-raw" in slug or "-kr" in slug:
            detected_lang = "ko"
        elif "-cn" in slug or "-zh" in slug:
            detected_lang = "zh"
        elif "-jp" in slug:
            detected_lang = "ja"

        logger.info("comix.to: %s — %d chapters (%s)", slug, len(chapters), detected_lang)
        return SourceInfo(suggested_title=title, suggested_lang=detected_lang, chapters=chapters)

    async def _scrape_from_dom(self, url: str, slug: str) -> list[DiscoveredChapter]:
        """Scrape chapter links from rendered DOM via browser CDP.

        Navigates to the title page, clicks through all pagination pages,
        and scrapes chapter links from the DOM.
        """
        import asyncio

        all_chapters: dict[float, list[str]] = defaultdict(list)

        # Navigate to title page
        result = await self._browser.execute_js(
            f'window.location.href = "{url}"',
            _DOMAIN, timeout=30
        )
        await asyncio.sleep(3)  # Wait for initial render

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
                        text: a.textContent.trim()
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

            logger.debug("DOM page %d: %d links, %d new unique", page_target, len(dom_chapters), new_count)

            # Try to click Next page
            click_js = f'''
            (() => {{
                // Look for a page link to page {page_target + 1}
                const next = document.querySelector('.page-link[href="#{page_target + 1}"]');
                if (next) {{
                    next.click();
                    return 'clicked page {page_target + 1}';
                }}
                // Or try the "Next" button
                const nextBtn = [...document.querySelectorAll('.page-link')].find(a => a.textContent.trim() === 'Next');
                if (nextBtn && !nextBtn.closest('.disabled')) {{
                    nextBtn.click();
                    return 'clicked Next';
                }}
                return 'no more pages';
            }})()
            '''
            result = await self._browser.execute_js(click_js, _DOMAIN, timeout=15)
            click_result = result.get("value", "")

            if "no more" in click_result or new_count == 0 and page_target > 1:
                break

            await asyncio.sleep(3)  # Wait for React to update DOM
            page_target += 1

            # Safety: limit to 50 pages
            if page_target > 50:
                break

        # Build DiscoveredChapter list
        chapters = []
        for num in sorted(all_chapters):
            variants = [
                ChapterVariant(
                    id=f"ch{num}_{i}",
                    url=href,
                )
                for i, href in enumerate(all_chapters[num])
            ]
            chapters.append(DiscoveredChapter(number=num, variants=variants))

        return chapters

    async def _discover_via_api(self, prefix: str, slug: str) -> list[DiscoveredChapter]:
        """Fallback: discover via API (needs anti-CSRF token, often fails)."""
        all_items: list[dict] = []
        page = 1
        while True:
            try:
                data = await self._fetch_json(
                    f"{_BASE}/api/v2/manga/{prefix}/chapters?limit=100&page={page}&order[number]=desc"
                )
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
            variants = [
                ChapterVariant(
                    id=str(item.get("chapter_id", f"ch{num}_{i}")),
                    url=f"{_BASE}/title/{slug}/{item.get('chapter_id', '')}-chapter-{num}",
                    group=(item.get("scanlation_group") or {}).get("name"),
                    votes=item.get("votes", 0),
                )
                for i, item in enumerate(by_number[num])
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
