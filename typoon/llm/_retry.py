"""Provider-agnostic retry helper for transient API failures.

Each provider's SDK has its own exception hierarchy, but the retry
policy is the same: exponential backoff with jitter for 429 and 5xx,
honoring server-supplied Retry-After when available. This module
factors that policy out so anthropic/gemini/openai don't drift.

Usage:

    resp = await with_retry(
        lambda: client.do_thing(...),
        is_retryable=_is_retryable,
        parse_retry_after=_parse_retry_after,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Tuned for paid Tier 1 providers under bursty fan-out (translate
# stage gathers windows in parallel). Five attempts gives the worst-
# case wait of ~1.5 + 3 + 6 + 12 + 24 ≈ 47s before failing the chapter.
MAX_RETRIES = 5
BASE_BACKOFF = 1.5     # seconds; doubled each attempt + jitter
MAX_BACKOFF = 30.0     # seconds; cap for the local exponential
RETRY_AFTER_CAP = 60.0  # seconds; ignore server hints longer than this


def parse_retry_after_header(headers) -> float | None:
    """Best-effort Retry-After parse out of a response headers mapping.

    Returns seconds (float) or None. Accepts the seconds form only;
    HTTP-date is rare for 429 in our flow and we'd rather backoff than
    block on a misparsed timestamp.
    """
    if not headers:
        return None
    raw = None
    # SDKs vary: openai uses lowercased Mapping, anthropic exposes the
    # underlying httpx Headers (case-insensitive), gemini hides the
    # response in google.api_core errors. All accept get().
    for key in ("retry-after", "Retry-After"):
        try:
            raw = headers.get(key)
        except Exception:  # pragma: no cover - defensive
            continue
        if raw:
            break
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    is_retryable: Callable[[BaseException], bool],
    parse_retry_after: Callable[[BaseException], float | None],
    provider: str,
    max_retries: int = MAX_RETRIES,
) -> T:
    """Run `call()` with exponential backoff for transient failures.

    `is_retryable(exc)` decides whether to retry; `parse_retry_after`
    extracts a server-supplied wait hint (or None). Non-retryable
    exceptions raise immediately. Hitting `max_retries` reraises the
    last exception.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except BaseException as exc:
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            last_exc = exc
            await _sleep_backoff(attempt, exc, parse_retry_after, provider)
    # Defensive: the loop above either returns or raises.
    assert last_exc is not None
    raise last_exc  # pragma: no cover


async def _sleep_backoff(
    attempt: int,
    exc: BaseException,
    parse_retry_after: Callable[[BaseException], float | None],
    provider: str,
) -> None:
    server_hint = parse_retry_after(exc)
    if server_hint is not None:
        delay = min(server_hint, RETRY_AFTER_CAP)
        source = f"server Retry-After={server_hint:.1f}s"
    else:
        delay = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
        delay += random.uniform(0, delay * 0.25)
        source = f"backoff attempt={attempt + 1}"
    logger.warning(
        "%s provider retry: %s (%s) — sleeping %.1fs",
        provider, type(exc).__name__, source, delay,
    )
    await asyncio.sleep(delay)
