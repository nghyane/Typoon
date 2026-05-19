"""Lens HTTP call with retry + exponential backoff.

The chrome-lens-py client surfaces network and rate-limit failures as
plain exceptions. Production usage (multi-chapter parallel) hits Google's
unpublished per-IP limits often enough that a silent ``return []`` drops
real text. Both tile-pass and bubble-pass route through this helper.

Strategy:
  - up to ``max_attempts`` tries (default 3)
  - exponential backoff: 1s, 2s, 4s, with full-jitter
  - log every retry; only raise the final exception
  - caller still wraps in try/except to preserve its drop-on-fail semantics
"""

from __future__ import annotations

import asyncio
import logging
import random


__all__ = ["lens_call_with_retry"]

logger = logging.getLogger(__name__)


async def lens_call_with_retry(
    api,
    image,
    *,
    ocr_language: str = "",
    output_format: str = "detailed",
    max_attempts: int = 3,
    base_delay: float = 1.0,
    label: str = "lens",
) -> dict:
    """Call ``api.process_image`` with exponential backoff.

    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await api.process_image(
                image, ocr_language=ocr_language, output_format=output_format,
            )
        except Exception as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt) + random.random() * base_delay
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                label, attempt + 1, max_attempts, e, delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
