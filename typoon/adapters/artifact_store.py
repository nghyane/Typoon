"""ArtifactStore — public-facing variant of BlobStore.

Adds `url(locator, version)` for browser fetch. Used for the rendered
chapter archive (`render.bnl`) which is the only artifact a user
consumes. Pipeline blobs (prepared, masks) use BlobStore directly —
they never get a public URL.

Implementations:

  LocalArtifactStore        wraps LocalBlobStore + serves via FastAPI
                            /files static mount (dev only)
  HuggingFaceArtifactStore  uploads to a public HF dataset; URLs go
                            through the bunle CDN /t/ route

Multi-backend coexistence: each chapter row stores `archive_backend` +
`archive_locator`, so chapters rendered against an old backend keep
serving after the operator switches the primary writer.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Protocol

from typoon.adapters.blob_store import BlobStore, LocalBlobStore


class ArtifactStore(BlobStore, Protocol):
    """BlobStore that also exposes a public URL for browser fetch."""

    def url(self, locator: str, *, version: str = "") -> str:
        """Return a public URL for the locator. Version is appended as
        a query string (`?v=…`) to bust CDN cache on re-renders without
        changing the cache-key path."""
        ...


# ── Local — disk + FastAPI StaticFiles ────────────────────────────────


class LocalArtifactStore(LocalBlobStore):
    """LocalBlobStore + URL via FastAPI `/files` static mount.

    Browser fetches resolve through the same FastAPI app at
    `/files/<locator>` (relative path). In production the SPA and API
    share the same origin via Cloudflare Tunnel URL Mappings, so a
    relative path works. In dev the Vite proxy forwards `/files` to
    the API.

    Use only for dev or single-host deploys; in multi-host the local
    artifact store on each worker is private to that host.
    """

    backend_name = "local"
    _MOUNT = "/files"

    def __init__(self, root: Path) -> None:
        super().__init__(root)

    def url(self, locator: str, *, version: str = "") -> str:
        qs = f"?v={version}" if version else ""
        return f"{self._MOUNT}/{locator}{qs}"


# ── HuggingFace — public dataset + bunle CDN ──────────────────────────


# Wall-clock cap for a single upload_file call. Sized for typical
# rendered chapter archives (5-30 MB) on a ~50 Mbit home upstream:
# the math says <2 min easily, so 10 min covers slow connections +
# HF Hub retries while still bounding the worst case so a half-dead
# TCP socket can't park a chapter "running" forever.
_PUT_TIMEOUT = 600.0


class HuggingFaceArtifactStore:
    """Public read via bunle CDN, write via HF Hub.

    Files land at `<path_prefix>/<key>` inside the configured HF dataset
    repo. The bunle CDN's `/t/` route proxies that path with edge cache,
    Range support, and CORS — this class never serves bytes itself.
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

        # `huggingface_hub.upload_file` is a sync HTTP call wrapping a
        # multi-part POST against `hf.co`. We have observed it stall
        # silently at >80% upload when the underlying connection goes
        # half-dead (Cloudflare keeps the TCP socket open but no bytes
        # flow through). Without a wall-clock cap the render worker
        # blocks indefinitely on a single chapter, the `tasks` claim
        # never releases, and the UI sees the chapter "stuck running"
        # with no way to retry. Cap each upload at `_PUT_TIMEOUT`; on
        # expiry asyncio cancels the executor task and the stage
        # runner classifies the failure as transient (requeue) via
        # `_run_one`'s exception path.
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    upload_file,
                    path_or_fileobj=str(src),
                    path_in_repo=self._hf_path(key),
                    repo_id=self._repo,
                    repo_type="dataset",
                    token=self._token,
                ),
                timeout=_PUT_TIMEOUT,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"huggingface upload timeout after {_PUT_TIMEOUT:.0f}s "
                f"(key={key}, size={src.stat().st_size} bytes)"
            ) from e
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

    async def exists(self, locator: str) -> bool:
        from huggingface_hub import HfFileSystem
        try:
            fs = HfFileSystem(token=self._token)
            return await asyncio.to_thread(
                fs.exists,
                f"datasets/{self._repo}/{self._hf_path(locator)}",
            )
        except Exception:
            return False

    async def aclose(self) -> None:
        return None

    def url(self, locator: str, *, version: str = "") -> str:
        qs = f"?v={version}" if version else ""
        return f"{self._cdn_prefix}/{locator}{qs}"
