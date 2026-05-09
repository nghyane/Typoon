"""Google Lens OCR — page-level via `chrome-lens-py`.

Google Lens reverse-engineered API. Best raw quality across stylised
manhwa / manga fonts in our benchmarks; downside is a network round
trip per tile and dependence on an unofficial endpoint.

Lens resizes any input over ~1000px on the longest axis, so we tile
every page into 720×900 windows that fit the limit without resampling.
Tiles overlap by `_OVERLAP` pixels — wider than the largest plausible
text block — so a bubble straddling a tile boundary appears intact in
at least one tile and survives dedup.

Concurrency: `chrome-lens-py` runs requests through `httpx.AsyncClient`
gated by an internal semaphore. We saw stable speed-ups up to
`max_concurrent=15`; beyond that variance jumps as the endpoint starts
shedding load. Keep at 15.

Output format: `'blocks'` returns whole bubbles already segmented (one
text block per dialog bubble), so we skip line/word reconstruction and
work at block granularity directly.

Dedup across tiles: blocks from overlap regions return either as
identical text on near-identical bboxes, or as a partial/truncated copy
nested inside a longer block. We sort by text length descending and
drop anything whose bbox sits ≥50% inside an already-kept block, plus
shorter substrings that overlap any kept block — handles both cases
without text comparison heuristics.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .types import Observation


logger = logging.getLogger(__name__)

# Upstream Lens endpoint. The Discord Activity proxy
# (`https://<application_id>.discordsays.com/lens/v1/crupload`) routes
# to the same destination but goes via Cloudflare's edge — measured
# 30–50% faster from APAC than the direct path and has noticeably
# lower variance under load. Override the endpoint by setting
# `LENS_ENDPOINT` (env) or constructor `endpoint` arg.
_DEFAULT_ENDPOINT = "https://lensfrontend-pa.googleapis.com/v1/crupload"

# Lens resizes images longer than ~1000px on the longest axis. 900 keeps a
# margin for the source aspect ratio without resampling.
_TILE_H = 900
_OVERLAP = 200
_MAX_CONCURRENT = 15
_DEDUP_IOU = 0.5
_SUBSTRING_IOU = 0.05


def is_available() -> bool:
    try:
        import chrome_lens_py   # noqa: F401
        return True
    except ImportError:
        return False


_LANG_MAP: dict[str, str] = {
    "en":    "en",
    "ja":    "ja",
    "ko":    "ko",
    "zh":    "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "vi":    "vi",
}


class GoogleLensPageOcr:
    """`chrome-lens-py`-backed page OCR with vertical tiling."""

    def __init__(
        self,
        max_concurrent: int = _MAX_CONCURRENT,
        endpoint: str | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        # Endpoint priority: explicit > env > library default. Patching
        # the module constant rather than passing it through LensAPI
        # because `chrome-lens-py` reads the URL at module-import time
        # in core.request_handler — there is no public override.
        import os
        self._endpoint = endpoint or os.environ.get("LENS_ENDPOINT") or _DEFAULT_ENDPOINT
        self._api: object | None = None

    def _get_api(self):
        if self._api is None:
            self._patch_endpoint()
            from chrome_lens_py import LensAPI
            self._api = LensAPI(max_concurrent=self._max_concurrent)
        return self._api

    def _patch_endpoint(self) -> None:
        """Repoint chrome-lens-py at the configured endpoint.

        Idempotent: only reloads `request_handler` if the constant
        actually changed, so repeated `GoogleLensPageOcr()` instances
        don't churn the import system.
        """
        import importlib
        import chrome_lens_py.constants as constants
        if constants.LENS_CRUPLOAD_ENDPOINT == self._endpoint:
            return
        constants.LENS_CRUPLOAD_ENDPOINT = self._endpoint
        from chrome_lens_py.core import request_handler
        importlib.reload(request_handler)

    def ocr_page(
        self,
        image: np.ndarray,
        *,
        lang: str | None = None,
    ) -> list[Observation]:
        if image.size == 0:
            return []
        ocr_lang = _LANG_MAP.get((lang or "en").lower(), "en")
        coro = self._ocr_async(image, ocr_lang)
        # The vision pipeline is synchronous but its callers (CLI, workers)
        # often run inside an asyncio event loop. `asyncio.run` would raise;
        # instead spin up a one-shot loop on a worker thread so the request
        # is still completed. When called from a sync context (no running
        # loop) we use `asyncio.run` directly.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def _ocr_async(self, image: np.ndarray, ocr_lang: str) -> list[Observation]:
        api = self._get_api()
        height, width = image.shape[:2]
        step = _TILE_H - _OVERLAP

        tiles: list[tuple[int, np.ndarray]] = []
        y = 0
        while y < height:
            y_end = min(y + _TILE_H, height)
            if y_end - y < 100:
                break
            tiles.append((y, image[y:y_end].copy()))
            if y_end == height:
                break
            y += step

        async def ocr_one(origin_y: int, tile: np.ndarray) -> list[Observation]:
            return await _ocr_tile(api, tile, origin_y, width, ocr_lang)

        results = await asyncio.gather(*(ocr_one(o, t) for o, t in tiles))
        observations = [obs for batch in results for obs in batch]
        return _dedup(observations)


async def _ocr_tile(
    api: object,
    tile: np.ndarray,
    origin_y: int,
    page_width: int,
    ocr_lang: str,
) -> list[Observation]:
    try:
        result = await api.process_image(  # type: ignore[attr-defined]
            tile,
            ocr_language=ocr_lang,
            output_format="blocks",
        )
    except Exception as e:
        logger.warning("google-lens tile OCR failed at y=%d: %s", origin_y, e)
        return []

    tile_h = tile.shape[0]
    out: list[Observation] = []
    for block in result.get("text_blocks", []) or []:
        text = (block.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        geom = block.get("geometry") or {}
        cx = geom.get("center_x", 0.0) * page_width
        cy = geom.get("center_y", 0.0) * tile_h
        w = geom.get("width", 0.0) * page_width
        h = geom.get("height", 0.0) * tile_h
        x1 = max(0, int(cx - w / 2))
        x2 = min(page_width, int(cx + w / 2))
        y1 = origin_y + int(cy - h / 2)
        y2 = origin_y + int(cy + h / 2)
        # Lens does not surface a per-block confidence — observations from a
        # successful response are equally trusted.
        out.append(Observation(bbox=(x1, y1, x2, y2), text=text, confidence=1.0))
    return out


def _dedup(observations: list[Observation]) -> list[Observation]:
    """Spatial + substring dedup across tile overlaps.

    Sorted by descending text length so the longest version of a block wins
    when a partial copy spans into the overlap region.
    """
    sorted_obs = sorted(observations, key=lambda o: -len(o.text))
    kept: list[Observation] = []
    for obs in sorted_obs:
        keep = True
        for k in kept:
            iou_self = _iou_self(obs.bbox, k.bbox)
            if iou_self > _DEDUP_IOU:
                keep = False
                break
            if iou_self > _SUBSTRING_IOU and obs.text in k.text:
                keep = False
                break
        if keep:
            kept.append(obs)
    return kept


def _iou_self(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection area divided by `a`'s area — measures how much of `a` is inside `b`."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area
