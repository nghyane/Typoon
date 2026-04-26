"""Opaque translation key generation."""

from __future__ import annotations

import hashlib
import json

from typoon.domain.bubble import Bubble

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def assign_keys(bubbles: list[Bubble], *, project_id: int, chapter: float) -> dict[str, Bubble]:
    """Assign stable opaque keys to bubbles and return key -> bubble."""
    out: dict[str, Bubble] = {}
    used: set[str] = set()
    for b in bubbles:
        salt = 0
        while True:
            key = _make_key(b, project_id=project_id, chapter=chapter, salt=salt)
            if key not in used:
                break
            salt += 1
        used.add(key)
        b.translation_key = key
        out[key] = b
    return out


def _make_key(b: Bubble, *, project_id: int, chapter: float, salt: int) -> str:
    payload = {
        "project": project_id,
        "chapter": chapter,
        "page": b.page_index,
        "idx": b.idx,
        "text": " ".join(b.source_text.split()),
        "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in b.polygon],
        "salt": salt,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    n = int.from_bytes(hashlib.blake2s(raw, digest_size=5).digest(), "big")
    chars = []
    for _ in range(7):
        n, r = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[r])
    return "".join(chars)
