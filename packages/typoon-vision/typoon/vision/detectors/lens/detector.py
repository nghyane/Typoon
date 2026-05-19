"""LensBlocksDetector — two-phase Lens OCR.

Phase A (tile_pass) localises text across the whole page coarsely.
Phase B (bubble_pass) re-OCRs DETR-anchored regions whose Phase-A
coverage is empty or partial, producing authoritative text inside
bubbles. The two phases share geometry projection and filter rules
from sibling modules.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

import numpy as np

from ...contracts import DetectionResult
from . import bubble_pass, filters, tile_pass


__all__ = ["LensBlocksDetector", "LensUnavailableError"]

logger = logging.getLogger(__name__)


_DEFAULT_ENDPOINT = "https://lensfrontend-pa.googleapis.com/v1/crupload"
_MAX_CONCURRENT = 15


# ─── Errors ───────────────────────────────────────────────────────────────


class LensUnavailableError(RuntimeError):
    """Raised when Lens dependency missing or upstream unreachable."""


# ─── Lang hint mapping ───────────────────────────────────────────────────


# English / unset stay on Lens auto-detect so mixed-script pages keep
# multiple scripts. Other source languages pass through as an explicit
# hint.
_LANG_HINTS: dict[str, str] = {
    "ja":      "ja",   "ja-JP":   "ja",
    "zh":      "zh-Hans", "zh-CN": "zh-Hans", "zh-Hans": "zh-Hans",
    "zh-Hant": "zh-Hant", "zh-TW": "zh-Hant", "zh-HK":   "zh-Hant",
    "ko":      "ko",   "ko-KR":   "ko",
    "vi":      "vi",   "vi-VN":   "vi",
}


def _lang_hint(source_lang: str | None) -> str:
    if not source_lang:
        return ""
    if source_lang.lower().startswith("en"):
        return ""
    return _LANG_HINTS.get(source_lang, "")


# ─── Detector ─────────────────────────────────────────────────────────────


class LensBlocksDetector:
    """Lens-as-detector with built-in recognition.

    Owns the chrome-lens-py client + the Comic-DETR side detector.
    The detect() method is the only public surface.
    """

    name = "lens_blocks"

    def __init__(
        self,
        *,
        comic_detr,
        endpoint: str | None = None,
        max_concurrent: int = _MAX_CONCURRENT,
        api: object | None = None,
    ) -> None:
        self._endpoint = (
            endpoint or os.environ.get("LENS_ENDPOINT") or _DEFAULT_ENDPOINT
        )
        self._max_concurrent = max_concurrent
        self._api: object | None = api      # may be injected by caller
        self._comic_detr = comic_detr

    async def detect(self, image: np.ndarray, lang: str | None) -> DetectionResult:
        api = await self._get_api()
        h, w = image.shape[:2]
        lang_hint = _lang_hint(lang)

        # Phase A + Comic-DETR run in parallel.
        try:
            async with asyncio.TaskGroup() as tg:
                tile_task = tg.create_task(tile_pass.run(api, image, lang_hint))
                detr_task = tg.create_task(
                    asyncio.to_thread(self._comic_detr.detect, image),
                )
        except* Exception as eg:
            raise LensUnavailableError(
                f"lens_blocks detector failed: {eg.exceptions[0]!r}"
            ) from eg

        coarse_blocks, detected_lang = tile_task.result()
        detr_dets = detr_task.result()
        bubble_regions = tuple(
            (d.cls, d.bbox, d.conf) for d in detr_dets
        )

        # Filter Phase-A noise before deciding what Phase B needs to redo.
        kept, rejected = filters.apply(coarse_blocks)

        # Phase B: re-OCR empty / partially-covered anchors.
        recovery_hint = lang_hint or _lang_hint(detected_lang)
        kept = await bubble_pass.run(
            api, image, kept, bubble_regions, recovery_hint,
        )

        return DetectionResult(
            blocks=tuple(kept),
            text_already_recognized=True,
            page_size=(w, h),
            rejected=tuple(rejected),
            detected_lang=detected_lang,
            bubble_mask=None,
            bubble_regions=bubble_regions,
        )

    # ─── chrome-lens-py setup ────────────────────────────────────────────

    async def _get_api(self):
        if self._api is None:
            self._patch_endpoint()
            try:
                from chrome_lens_py import LensAPI
            except ImportError as e:
                raise LensUnavailableError(
                    "chrome-lens-py not installed; install or switch pipeline preset"
                ) from e
            self._api = LensAPI(max_concurrent=self._max_concurrent)
        return self._api

    def _patch_endpoint(self) -> None:
        """Repoint chrome-lens-py at the configured endpoint (idempotent)."""
        try:
            constants = importlib.import_module("chrome_lens_py.constants")
        except ImportError:
            return
        if constants.LENS_CRUPLOAD_ENDPOINT == self._endpoint:
            return
        constants.LENS_CRUPLOAD_ENDPOINT = self._endpoint
        request_handler = importlib.import_module(
            "chrome_lens_py.core.request_handler",
        )
        importlib.reload(request_handler)
