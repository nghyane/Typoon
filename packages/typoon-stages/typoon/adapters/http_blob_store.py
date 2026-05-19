"""HttpBlobStore — pipeline blob storage over HTTP.

Reaches a Typoon node running the storage role (`typoon api --role
storage` or `--role full`) which exposes `/api/blobs/<key>` for
PUT/GET/HEAD/DELETE. Authenticates with a long-lived API token whose
scope includes `worker` (see typoon/api/auth_token.py).

The transport is plain HTTP. In production, all participating nodes
join the same Tailscale tailnet so blob traffic stays on the encrypted
mesh; the server binds 0.0.0.0 and is firewalled to the tailnet.

Idempotent: PUT replaces existing blobs. The pipeline writes and reads
deterministic keys (HMAC tokens of project/chapter), so concurrent
writes of the same key produce identical bytes.
"""

from __future__ import annotations

from pathlib import Path

import httpx


class HttpBlobStore:
    backend_name = "http"

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def put(self, key: str, src: Path) -> str:
        # Stream the file body so workers don't load multi-MB blobs into
        # RAM. We wrap the sync file in an async byte stream because
        # httpx.AsyncClient rejects sync iterables.
        size = src.stat().st_size
        with src.open("rb") as f:
            r = await self._client.put(
                self._url(key),
                content=_aiter_chunks(f),
                headers={
                    **self._headers,
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                },
            )
        if r.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{r.status_code} from {self._url(key)}: {r.text[:200]}",
                request=r.request, response=r,
            )
        return key

    async def get(self, locator: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with self._client.stream(
            "GET", self._url(locator), headers=self._headers,
        ) as r:
            r.raise_for_status()
            tmp = dest.with_name(f"{dest.name}.part")
            try:
                with tmp.open("wb") as f:
                    async for chunk in r.aiter_bytes(64 * 1024):
                        f.write(chunk)
                tmp.replace(dest)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    async def delete(self, locator: str) -> bool:
        r = await self._client.delete(self._url(locator), headers=self._headers)
        return r.status_code == 204

    async def exists(self, locator: str) -> bool:
        r = await self._client.head(self._url(locator), headers=self._headers)
        return r.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _url(self, key: str) -> str:
        return f"{self._base}/api/blobs/{key}"


def _iter_chunks(fh, size: int = 64 * 1024):
    while True:
        chunk = fh.read(size)
        if not chunk:
            break
        yield chunk


async def _aiter_chunks(fh, size: int = 64 * 1024):
    while True:
        chunk = fh.read(size)
        if not chunk:
            break
        yield chunk
