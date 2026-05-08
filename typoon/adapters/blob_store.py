"""BlobStore — opaque server-side blob storage for pipeline artifacts.

A BlobStore moves bytes between workers. It is NOT browser-facing — there
is no `url()` method and no expectation of edge caching. Used for
intermediate state (prepared.bnl, masks.npz) that flows through the
pipeline but never reaches a user.

Implementations:

  LocalBlobStore        filesystem under a root; same disk as caller
  HttpBlobStore         remote node reached via /api/blobs/* over HTTP
                        (typically tailnet); auth via worker API token

`ArtifactStore` (in artifact_store.py) is the public-facing variant —
extends BlobStore with `url()` for browser fetch.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    """Pipeline blob storage. No public URL — server-only."""

    backend_name: str

    async def put(self, key: str, src: Path) -> str:
        """Upload `src` to `key`. Returns the locator string the store
        will accept on subsequent get/delete calls. Path-based stores
        return the key itself; opaque-id stores may return a different
        token. Idempotent — re-uploading the same key overwrites."""
        ...

    async def get(self, locator: str, dest: Path) -> None:
        """Download blob into `dest`. Caller owns dest's parent dir."""
        ...

    async def delete(self, locator: str) -> bool:
        """Best-effort delete. Returns True if the blob existed."""
        ...

    async def exists(self, locator: str) -> bool:
        """Cheap presence check; used for skip-if-exists optimizations."""
        ...

    async def aclose(self) -> None:
        """Release any pooled resources (HTTP clients, etc.).
        Local stores can no-op."""
        ...


# ── Local impl ────────────────────────────────────────────────────────


class LocalBlobStore:
    """Filesystem-backed BlobStore.

    Keys map directly under `root`; `..` and absolute paths are rejected.
    Used directly in single-host dev; in multi-host it sits behind the
    HTTP blob endpoint on the storage node.
    """

    backend_name = "local"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    # ── Path safety ────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid blob key: {key!r}")
        return self._root / key

    # ── BlobStore ───────────────────────────────────────────────────

    async def put(self, key: str, src: Path) -> str:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
        try:
            shutil.copyfile(src, tmp)
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return key

    async def get(self, locator: str, dest: Path) -> None:
        src = self._path(locator)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    async def delete(self, locator: str) -> bool:
        path = self._path(locator)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True

    async def exists(self, locator: str) -> bool:
        return self._path(locator).exists()

    async def aclose(self) -> None:
        return None
