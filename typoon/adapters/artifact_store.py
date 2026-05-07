"""ArtifactStore — opaque blob storage for chapter archives.

Two operations only: put and get. A queue worker writes archives to the
store and the next stage reads them; nothing else touches storage. Other
operations (delete, head, signed URL, listing) belong to the app/UI layer
and are added there when needed.

LocalArtifactStore writes to disk. R2ArtifactStore lands when SaaS does.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    async def put_file(self, key: str, src: Path) -> None: ...
    async def get_file(self, key: str, dest: Path) -> None: ...
    async def delete(self, key: str) -> bool: ...


class LocalArtifactStore:
    """File-based store under a single root."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid artifact key: {key!r}")
        return self._root / key

    async def put_file(self, key: str, src: Path) -> None:
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

    async def get_file(self, key: str, dest: Path) -> None:
        src = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    async def delete(self, key: str) -> bool:
        """Best-effort delete. Returns True if the key existed."""
        path = self._path(key)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True
