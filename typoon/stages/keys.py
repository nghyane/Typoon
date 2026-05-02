"""Opaque translation key generation — pure, no side effects."""

from __future__ import annotations

import hashlib
import json

from typoon.domain.scan import Bubble as ScannedBubble, BubbleKey

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def assign_keys(
    bubbles: list[ScannedBubble],
    *,
    project_id: int,
    chapter_id: int,
) -> list[BubbleKey]:
    """Return stable list of BubbleKey. Key is the single identity for LLM communication."""
    out: list[BubbleKey] = []
    used: set[str] = set()
    for b in bubbles:
        salt = 0
        while True:
            key = _make_key(b, project_id=project_id, chapter_id=chapter_id, salt=salt)
            if key not in used:
                break
            salt += 1
        used.add(key)
        out.append(BubbleKey(key=key, bubble=b))
    return out


def _make_key(b: ScannedBubble, *, project_id: int, chapter_id: int, salt: int) -> str:
    payload = {
        "project":    project_id,
        "chapter_id": chapter_id,
        "page":       b.page_index,
        "idx":        b.idx,
        "salt":       salt,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    n = int.from_bytes(hashlib.blake2s(raw, digest_size=5).digest(), "big")
    chars = []
    for _ in range(7):
        n, r = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[r])
    return "".join(chars)
