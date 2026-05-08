"""ArtifactStore — opaque blob storage for chapter archives.

Multi-backend by design. Every backend implements four operations:

  put(key, src) -> locator    persist file, return locator string
  get(locator, dest)          fetch file by locator
  delete(locator) -> bool     idempotent delete
  url(locator, *, version)    pure-function URL for browser fetch

`key` is the desired storage path/name. Path-based backends (Local, HF,
R2, S3) use it directly and return it as the locator. Opaque-id backends
(Google Drive, future) use it as a hint and return a file_id.

Stages and API never inspect locators — they just persist them on the
chapter row and pass them back to the matching reader on URL build.
This keeps URL build a pure-function string concat (no IO on hot path).

Browser reads are served exclusively via the bunle CDN
(https://bunle-cdn-16g.pages.dev/<prefix>/<locator>) so all backends
share one CORS+edge-cache configuration. The CDN routes by prefix:
  /t/   →  HuggingFaceArtifactStore  (HF dataset)
  /r/   →  R2 public bucket          (future)
  /d/   →  Google Drive file_id      (future)
LocalArtifactStore returns app-relative `/files/<key>` for dev only.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    backend_name: str   # short identifier persisted on chapter row

    async def put(self, key: str, src: Path) -> str: ...
    async def get(self, locator: str, dest: Path) -> None: ...
    async def delete(self, locator: str) -> bool: ...
    def url(self, locator: str, *, version: str = "") -> str: ...


# ── Local — disk + FastAPI StaticFiles ────────────────────────────────


class LocalArtifactStore:
    """File-based store served by the FastAPI `/files` mount.

    Locator == key (path). Use for dev or for server-only artifacts
    (prepared, masks) whose bytes never leave the API host.
    """

    backend_name = "local"

    def __init__(self, root: Path, *, public_base: str = "/files") -> None:
        self._root = Path(root)
        self._public_base = public_base.rstrip("/")

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid artifact key: {key!r}")
        return self._root / key

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

    def url(self, locator: str, *, version: str = "") -> str:
        qs = f"?v={version}" if version else ""
        return f"{self._public_base}/{locator}{qs}"


# ── HuggingFace — public dataset + bunle CDN ──────────────────────────


class HuggingFaceArtifactStore:
    """Public read via bunle CDN, write via HF Hub.

    Files land at `<path_prefix>/<key>` inside the configured HF dataset
    repo. The bunle CDN's `/t/` route proxies that path with edge cache,
    Range support, and CORS — this class never serves bytes itself.

    Locator == key (path inside the typoon namespace, without
    `path_prefix`). The class joins the prefix on every call so the
    locator stored in DB stays portable if `path_prefix` ever changes.
    """

    backend_name = "huggingface"

    def __init__(
        self,
        *,
        repo: str,
        token: str,
        cdn_prefix: str,
        path_prefix: str = "typoon",
    ) -> None:
        self._repo = repo
        self._token = token
        self._cdn_prefix = cdn_prefix.rstrip("/")
        self._path_prefix = path_prefix.strip("/")

    def _hf_path(self, key: str) -> str:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid artifact key: {key!r}")
        return f"{self._path_prefix}/{key}"

    async def put(self, key: str, src: Path) -> str:
        from huggingface_hub import upload_file
        await asyncio.to_thread(
            upload_file,
            path_or_fileobj=str(src),
            path_in_repo=self._hf_path(key),
            repo_id=self._repo,
            repo_type="dataset",
            token=self._token,
        )
        return key

    async def get(self, locator: str, dest: Path) -> None:
        from huggingface_hub import hf_hub_download
        downloaded = await asyncio.to_thread(
            hf_hub_download,
            repo_id=self._repo,
            filename=self._hf_path(locator),
            repo_type="dataset",
            token=self._token,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, dest)

    async def delete(self, locator: str) -> bool:
        from huggingface_hub import delete_file
        try:
            await asyncio.to_thread(
                delete_file,
                path_in_repo=self._hf_path(locator),
                repo_id=self._repo,
                repo_type="dataset",
                token=self._token,
            )
            return True
        except Exception:
            return False

    def url(self, locator: str, *, version: str = "") -> str:
        qs = f"?v={version}" if version else ""
        return f"{self._cdn_prefix}/{locator}{qs}"


# ── Registry ──────────────────────────────────────────────────────────


class ArtifactStoreRegistry:
    """Holds one writer (primary) + one reader per known backend.

    The primary store handles all writes from worker pipelines. Reads
    dispatch by backend name persisted on the chapter row, so chapters
    rendered against one backend keep working after the operator
    switches the primary — no migration required.
    """

    __slots__ = ("_primary", "_by_name")

    def __init__(self, primary: ArtifactStore, all_stores: dict[str, ArtifactStore]) -> None:
        if primary.backend_name not in all_stores:
            all_stores = {**all_stores, primary.backend_name: primary}
        self._primary = primary
        self._by_name = all_stores

    @property
    def writer(self) -> ArtifactStore:
        return self._primary

    @property
    def primary_name(self) -> str:
        return self._primary.backend_name

    def reader(self, backend: str) -> ArtifactStore:
        try:
            return self._by_name[backend]
        except KeyError as e:
            raise RuntimeError(
                f"No artifact store configured for backend {backend!r}. "
                f"Available: {sorted(self._by_name)}",
            ) from e
