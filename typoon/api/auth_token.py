"""API token issuance and verification.

Format: `typ_<32 url-safe random chars>`. Plaintext shown ONCE at create
time, after that we keep only:
  - bcrypt hash (`token_hash`) — used for verify on each request.
  - first 8 chars (`prefix`) — used to narrow candidates and to display
    in the UI ("typ_AbCd…").

Lookup path on each request:
  1. Strip "Bearer ".
  2. Split prefix (chars 4..12 after "typ_") and remainder.
  3. Fetch active tokens with that prefix → almost always 0 or 1.
  4. bcrypt.checkpw against token_hash.
  5. On hit, fire-and-forget UPDATE last_used.

bcrypt work factor 10 → ~10ms verify. Phase 1 community small enough
that we don't need a verify cache.
"""

from __future__ import annotations

import asyncio
import logging
import secrets

import bcrypt

from typoon.storage import Store

logger = logging.getLogger(__name__)

_PREFIX = "typ_"
_PREFIX_LEN = 8        # chars after "typ_" used as DB index prefix
_BODY_LEN = 32         # url-safe random chars total
_BCRYPT_ROUNDS = 10


def _generate_token() -> tuple[str, str]:
    """Return (plaintext, prefix). Plaintext starts with `typ_`."""
    body = secrets.token_urlsafe(_BODY_LEN)[:_BODY_LEN]
    plaintext = _PREFIX + body
    return plaintext, body[:_PREFIX_LEN]


def _hash(plaintext: str) -> str:
    return bcrypt.hashpw(
        plaintext.encode("utf-8"),
        bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
    ).decode("utf-8")


def looks_like_api_token(raw: str) -> bool:
    return raw.startswith(_PREFIX)


async def issue_api_token(
    db: Store, *, user_id: int, name: str,
) -> tuple[int, str, str]:
    """Create a token row and return (id, plaintext, prefix).

    The plaintext is the only time the caller can see it. Caller is
    responsible for showing it to the user once and then discarding.
    """
    plaintext, prefix = _generate_token()
    token_hash = _hash(plaintext)
    token_id = await db.create_api_token(
        user_id=user_id, name=name, prefix=prefix, token_hash=token_hash,
    )
    return token_id, plaintext, prefix


async def verify_api_token(db: Store, raw: str) -> dict | None:
    """Look up an API token. Returns the user dict on hit, None on miss.

    On hit, schedules a background `touch_api_token` so the request is
    not blocked by an UPDATE.
    """
    if not looks_like_api_token(raw):
        return None
    body = raw[len(_PREFIX):]
    if len(body) < _PREFIX_LEN:
        return None
    prefix = body[:_PREFIX_LEN]

    candidates = await db.candidates_by_prefix(prefix)
    if not candidates:
        return None

    raw_bytes = raw.encode("utf-8")
    for cand in candidates:
        try:
            ok = bcrypt.checkpw(raw_bytes, cand["token_hash"].encode("utf-8"))
        except ValueError:
            logger.warning("bcrypt rejected hash for token id=%s", cand["id"])
            continue
        if ok:
            user = await db.get_user(cand["user_id"])
            if user is None:
                return None
            # Fire-and-forget last_used bump. Don't await — keeps
            # request latency at the bcrypt verify cost only.
            try:
                asyncio.create_task(db.touch_api_token(cand["id"]))
            except RuntimeError:
                # No running loop in some test contexts — skip silently.
                pass
            return user
    return None
