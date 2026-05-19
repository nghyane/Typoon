"""keys.py — port of typoon/stages/keys.py.

Deterministic 7-char base32 key from (chapter_id, page_index, idx, salt).
blake2s digest — same algorithm as the TypeScript Worker implementation.
"""
from __future__ import annotations
import hashlib
import json

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def make_key(chapter_id: str, page_index: int, idx: int, salt: int = 0) -> str:
    payload = json.dumps(
        {"chapter_id": chapter_id, "idx": idx, "page": page_index, "salt": salt},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    n = int.from_bytes(hashlib.blake2s(payload, digest_size=5).digest(), "big")
    chars = []
    for _ in range(7):
        n, r = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[r])
    return "".join(chars)


def assign_keys(chapter_id: str, page_index: int, count: int,
                used: set[str]) -> list[str]:
    keys = []
    for idx in range(count):
        salt = 0
        while True:
            k = make_key(chapter_id, page_index, idx, salt)
            if k not in used:
                break
            salt += 1
        used.add(k)
        keys.append(k)
    return keys
