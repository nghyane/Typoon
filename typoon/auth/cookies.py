"""Cookie store — per-domain cookie persistence for connectors.

Storage: TYPOON_HOME/cache/cookies.json

    {
      "comix.to": {
        "value": "cf_clearance=abc; __cf_bm=xyz",
        "updated": "2026-03-21T10:30:00Z"
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class CookieStore:
    """Per-domain cookie store. JSON file at TYPOON_HOME/cache/cookies.json."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from ..config import load_config
            _, paths = load_config()
            path = paths.cache / "cookies.json"
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def get(self, domain: str) -> str | None:
        entry = self._data.get(domain)
        return entry["value"] if entry and entry.get("value") else None

    def put(self, domain: str, cookies: str) -> None:
        self._data[domain] = {
            "value": cookies,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def has(self, domain: str) -> bool:
        return self.get(domain) is not None

    def remove(self, domain: str) -> None:
        self._data.pop(domain, None)
        self._save()

    def list_domains(self) -> list[dict]:
        return [
            {"domain": d, "updated": e.get("updated", "")}
            for d, e in self._data.items()
            if e.get("value")
        ]
