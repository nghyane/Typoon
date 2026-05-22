"""keys.py — opaque translation key generation.

Must match `workers/shared/src/keys.ts` exactly (same alphabet, same JSON
shape, same blake2s digest size). The TS pipeline re-derives keys from
(job_id, page_index, idx) at translate time and compares against the keys
written into the scan msgpack here — any divergence breaks brief noise
filtering and SFX routing silently.

Payload (sort_keys=True ⇒ idx, job_id, page, salt):
    {"idx": int, "job_id": int, "page": int, "salt": int}

job_id is an int on both sides so json.dumps emits it unquoted.
"""
from __future__ import annotations
import hashlib
import json

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def make_key(job_id: int, page_index: int, idx: int, salt: int = 0) -> str:
    payload = json.dumps(
        {"idx": idx, "job_id": job_id, "page": page_index, "salt": salt},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    n = int.from_bytes(hashlib.blake2s(payload, digest_size=5).digest(), "big")
    chars = []
    for _ in range(7):
        n, r = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[r])
    return "".join(chars)


def assign_keys(job_id: int, page_index: int, count: int,
                used: set[str]) -> list[str]:
    keys = []
    for idx in range(count):
        salt = 0
        while True:
            k = make_key(job_id, page_index, idx, salt)
            if k not in used:
                break
            salt += 1
        used.add(k)
        keys.append(k)
    return keys
