"""Pipeline blob storage HTTP endpoint.

Mounted only on nodes running the storage role. Workers reach this
through `HttpBlobStore` to share intermediate artifacts (prepared.bnl,
masks.npz) across hosts. Requires an API token with the `worker`
scope; ordinary user JWTs are rejected.

Endpoints (all under /api/blobs/<key>):

  PUT     idempotent upload, streams the body to disk
  GET     download with HTTP Range support (FileResponse handles it)
  HEAD    cheap presence check
  DELETE  best-effort delete

Keys are arbitrary path strings (the pipeline picks them via
`adapters.chapter_archive`); `..` and absolute paths are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from typoon.api.deps import get_paths, require_worker
from typoon.paths import Paths

router = APIRouter(prefix="/api/blobs", tags=["blobs"])


def _is_unsafe_key(key: str) -> bool:
    return key.startswith("/") or ".." in Path(key).parts


def _resolve(paths: Paths, key: str) -> Path:
    if _is_unsafe_key(key):
        raise HTTPException(400, "invalid blob key")
    return paths.artifacts / key


@router.put("/{key:path}", status_code=204)
async def put_blob(
    key: str,
    request: Request,
    paths: Paths = Depends(get_paths),
    user: dict = Depends(require_worker),
) -> Response:
    """Stream the request body into the blob store.

    Writes through a `.part` sibling and renames atomically so a
    crashed/cancelled upload never leaves a half-written file at the
    final path.
    """
    dest = _resolve(paths, key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.part.{os.getpid()}")
    try:
        async with aiofiles.open(tmp, "wb") as f:
            async for chunk in request.stream():
                await f.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return Response(status_code=204)


@router.get("/{key:path}")
async def get_blob(
    key: str,
    paths: Paths = Depends(get_paths),
    user: dict = Depends(require_worker),
):
    path = _resolve(paths, key)
    if not path.exists():
        raise HTTPException(404, "blob not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.head("/{key:path}")
async def head_blob(
    key: str,
    paths: Paths = Depends(get_paths),
    user: dict = Depends(require_worker),
) -> Response:
    path = _resolve(paths, key)
    return Response(status_code=200 if path.exists() else 404)


@router.delete("/{key:path}", status_code=204)
async def delete_blob(
    key: str,
    paths: Paths = Depends(get_paths),
    user: dict = Depends(require_worker),
) -> Response:
    path = _resolve(paths, key)
    path.unlink(missing_ok=True)
    return Response(status_code=204)
