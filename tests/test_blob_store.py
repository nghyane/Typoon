"""Tests for adapters.blob_store and adapters.http_blob_store."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from typoon.adapters.blob_store import LocalBlobStore
from typoon.adapters.http_blob_store import HttpBlobStore


# ── LocalBlobStore round-trip ────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_blob_store_roundtrip(tmp_path):
    store = LocalBlobStore(tmp_path / "store")
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello blob")

    locator = await store.put("nested/key.bin", src)
    assert locator == "nested/key.bin"

    assert await store.exists(locator)

    out = tmp_path / "out.bin"
    await store.get(locator, out)
    assert out.read_bytes() == b"hello blob"

    assert await store.delete(locator)
    assert not await store.exists(locator)
    assert not await store.delete(locator)


@pytest.mark.asyncio
async def test_local_blob_store_rejects_unsafe_keys(tmp_path):
    store = LocalBlobStore(tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")

    with pytest.raises(ValueError):
        await store.put("/abs/path", src)
    with pytest.raises(ValueError):
        await store.put("../escape", src)


# ── HttpBlobStore round-trip against an in-process mock ──────────────


def _build_mock_app() -> tuple[object, dict[str, bytes]]:
    """Tiny in-process Starlette app that mimics /api/blobs/<key>.

    Returns (app, store_dict) so tests can assert on stored bytes.
    Starlette is used directly (not FastAPI) so route signatures stay
    straightforward — this avoids FastAPI mistaking the `request`
    argument for a query parameter under ASGITransport.
    """
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route

    store: dict[str, bytes] = {}

    async def put_blob(request):
        body = b""
        async for chunk in request.stream():
            body += chunk
        store[request.path_params["key"]] = body
        return Response(status_code=204)

    async def get_blob(request):
        key = request.path_params["key"]
        if key not in store:
            return Response(status_code=404)
        return Response(content=store[key], media_type="application/octet-stream")

    async def head_blob(request):
        return Response(status_code=200 if request.path_params["key"] in store else 404)

    async def delete_blob(request):
        store.pop(request.path_params["key"], None)
        return Response(status_code=204)

    app = Starlette(routes=[
        Route("/api/blobs/{key:path}", put_blob, methods=["PUT"]),
        Route("/api/blobs/{key:path}", get_blob, methods=["GET"]),
        Route("/api/blobs/{key:path}", head_blob, methods=["HEAD"]),
        Route("/api/blobs/{key:path}", delete_blob, methods=["DELETE"]),
    ])
    return app, store


@pytest.mark.asyncio
async def test_http_blob_store_roundtrip(tmp_path):
    app, server_state = _build_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver",
    ) as client:
        store = HttpBlobStore(
            base_url="http://testserver",
            api_token="t",
            client=client,
        )

        src = tmp_path / "src.bin"
        payload = b"binary content " * 256
        src.write_bytes(payload)

        locator = await store.put("render/abc.bnl", src)
        assert locator == "render/abc.bnl"
        assert server_state["render/abc.bnl"] == payload

        assert await store.exists("render/abc.bnl")

        out = tmp_path / "out.bin"
        await store.get("render/abc.bnl", out)
        assert out.read_bytes() == payload

        assert await store.delete("render/abc.bnl")
        assert not await store.exists("render/abc.bnl")
