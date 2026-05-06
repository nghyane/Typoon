"""ArtifactStore — opaque blob storage with the same logical keys for local
dev (`~/.typoon/artifacts/<key>`) and R2 (`r2://<bucket>/<key>`).

Stages and storage code never know whether they're talking to disk or R2.
Both implementations honor the same atomicity/idempotency contract.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ArtifactHead:
    size: int
    etag: str | None = None


class ArtifactStore(Protocol):
    async def put_file(self, key: str, src: Path) -> None: ...
    async def get_file(self, key: str, dest: Path) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def head(self, key: str) -> ArtifactHead | None: ...
    async def url(self, key: str, expires: int = 3600) -> str: ...


class LocalArtifactStore:
    """File-based store under a single root.

    Atomicity: put_file writes to `<key>.tmp.<pid>` then `os.replace` to `<key>`.
    delete is a no-op on missing keys.
    url returns `file://` paths for local-dev convenience; the API server
    serves them via a signed-URL wrapper in production.
    """

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

    async def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)

    async def head(self, key: str) -> ArtifactHead | None:
        path = self._path(key)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return ArtifactHead(size=stat.st_size, etag=None)

    async def url(self, key: str, expires: int = 3600) -> str:
        return self._path(key).resolve().as_uri()
