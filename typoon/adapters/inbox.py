"""ChapterInbox — short-lived storage for browser-uploaded chapter zips.

The inbox is the bridge between browser PUT and the engine's ingest
worker. Clients (web SPA, browser extension) PUT a chapter zip via
pre-signed URLs straight to the inbox backend, bypassing the engine's
home upstream bandwidth. The engine then fetches the zip from the
inbox, unpacks, prepares, and packs prepared.bnl.

Why a dedicated abstraction instead of reusing `BlobStore`?
  - BlobStore is internal: never browser-facing, no presigning.
  - The inbox is **temporary**: keys live minutes between PUT and
    ingest. The bucket should have a 24h lifecycle rule as a safety
    net for upload-aborted prefixes.
  - Multipart upload semantics (init / part presign / complete /
    abort) don't fit BlobStore's two-method protocol.

Multipart only — single PUT is too fragile at the chapter sizes we
see (>50 MB common). All zips, regardless of size, go through the
multipart flow. R2/S3 enforce min 5 MB per non-final part; we use
8 MB by default which is a good compromise between concurrency and
operation count (10× cheaper than per-page PUT).

Backends:

  S3Inbox     R2 / AWS S3 / Backblaze B2 / MinIO / Wasabi
              (anything S3-compatible — only the endpoint differs).
  LocalInbox  dev only — multipart simulated as files on disk.

The CORS rule on the S3 bucket MUST expose the `ETag` response header
so browser clients can read it from the PUT response and pass it back
to `complete_multipart`. Without it the browser sees `null` and
upload-finalize will reject every part.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import botocore.client

from typoon.config import StorageConfig

logger = logging.getLogger(__name__)

# Pre-signed URL TTL. Long enough that a ~100 MB chapter on a 5 Mbps
# upstream link (~3 minutes) finishes inside one window even with
# retries, but short enough that orphaned URLs in browser memory or
# storage decay quickly.
PRESIGN_TTL_SECONDS = 3600

# Default part size. R2/S3 require ≥ 5 MiB for non-final parts; 8 MiB
# is a good fit for typical chapter sizes (12 parts for 100 MB) and
# keeps PUT round-trip count low while letting the client stripe 4
# parts in flight at ~32 MB peak memory.
DEFAULT_PART_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class PartUrl:
    number: int   # 1-based, matches S3 PartNumber
    url:    str


@dataclass(frozen=True)
class CompletedPart:
    number: int
    etag:   str   # raw ETag value the client read from PUT response


@dataclass(frozen=True)
class InboxHandle:
    """Persisted handle for a deferred prepare task.

    Captures the multipart-upload coordinates the prepare worker needs
    to materialise the chapter zip on disk: tmp_id (R2 key prefix),
    upload_id (S3 multipart id), and the per-part ETag list. One handle
    per chapter; rows live in `chapter_inbox` until prepare completes.
    """
    chapter_id: int
    tmp_id:     str
    upload_id:  str
    parts:      tuple[CompletedPart, ...]
    title:      str | None = None


class ChapterInbox(Protocol):
    """Multipart-only inbox for chapter zips."""

    backend_name: str

    async def create_multipart(
        self, *, tmp_id: str, part_count: int, part_size: int,
    ) -> tuple[str, list[PartUrl]]:
        """Begin a multipart upload at `tmp_id`.

        Returns (upload_id, presigned_part_urls). Each URL accepts a
        single PUT of up to `part_size` bytes (the final part may be
        smaller). The client must record ETag from each PUT response
        and pass them to `complete_multipart`.
        """
        ...

    async def complete_multipart(
        self, *, tmp_id: str, upload_id: str, parts: list[CompletedPart],
    ) -> int:
        """Finalize the multipart upload. Returns the resulting object
        size in bytes. Raises on ETag mismatch or missing parts."""
        ...

    async def abort_multipart(
        self, *, tmp_id: str, upload_id: str,
    ) -> None:
        """Cancel an in-progress upload. Safe to call repeatedly. The
        bucket lifecycle rule sweeps orphans even if this is skipped."""
        ...

    async def fetch(self, *, tmp_id: str, dest: Path) -> int:
        """Stream the completed zip to `dest`. Returns bytes written."""
        ...

    async def delete(self, *, tmp_id: str) -> None:
        """Remove the completed zip after ingest. Best-effort."""
        ...


def is_configured(cfg: StorageConfig) -> bool:
    """True when the operator wired up a remote S3-compatible inbox.

    Local inbox does not require config — it falls through to the
    on-disk implementation rooted at `paths.artifacts/_inbox/`.
    """
    spec = cfg.inbox
    if spec.type == "local":
        return True
    if spec.type == "s3":
        return bool(
            spec.s3_endpoint
            and spec.s3_access_key_id
            and spec.s3_secret_access_key
            and spec.s3_bucket
        )
    return False


class S3Inbox:
    """S3-compatible inbox. Tested against R2; should also work
    against AWS S3, Backblaze B2, MinIO, Wasabi unchanged.

    Only the `endpoint_url` and credentials change between providers.
    Every method runs `boto3` (sync) inside `asyncio.to_thread` so
    the event loop stays responsive under concurrent uploads.
    """

    backend_name = "s3"

    def __init__(
        self,
        *,
        endpoint:   str,
        bucket:     str,
        region:     str,
        access_key: str,
        secret_key: str,
        prefix:     str = "tmp/",
    ) -> None:
        self._endpoint = endpoint
        self._bucket   = bucket
        self._region   = region
        self._access   = access_key
        self._secret   = secret_key
        self._prefix   = prefix.rstrip("/") + "/"
        self._client: "botocore.client.BaseClient | None" = None

    def _key(self, tmp_id: str) -> str:
        return f"{self._prefix}{tmp_id}.zip"

    def _client_for(self) -> "botocore.client.BaseClient":
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config as BotoConfig
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access,
            aws_secret_access_key=self._secret,
            region_name=self._region,
            config=BotoConfig(
                signature_version="s3v4",
                # Allow multiple parts + the engine's fetch to share
                # one keep-alive pool. 32 covers ~4 concurrent uploads
                # × 8 in-flight parts each plus headroom.
                max_pool_connections=32,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        return self._client

    async def create_multipart(
        self, *, tmp_id: str, part_count: int, part_size: int,
    ) -> tuple[str, list[PartUrl]]:
        client = self._client_for()
        key    = self._key(tmp_id)

        def _do() -> tuple[str, list[PartUrl]]:
            init = client.create_multipart_upload(
                Bucket=self._bucket, Key=key,
                ContentType="application/zip",
            )
            upload_id = init["UploadId"]
            urls: list[PartUrl] = []
            for n in range(1, part_count + 1):
                url = client.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self._bucket,
                        "Key":    key,
                        "UploadId":   upload_id,
                        "PartNumber": n,
                    },
                    ExpiresIn=PRESIGN_TTL_SECONDS,
                )
                urls.append(PartUrl(number=n, url=url))
            return upload_id, urls

        return await asyncio.to_thread(_do)

    async def complete_multipart(
        self, *, tmp_id: str, upload_id: str, parts: list[CompletedPart],
    ) -> int:
        client = self._client_for()
        key    = self._key(tmp_id)

        def _do() -> int:
            # S3 requires parts sorted by PartNumber.
            ordered = sorted(parts, key=lambda p: p.number)
            client.complete_multipart_upload(
                Bucket=self._bucket, Key=key, UploadId=upload_id,
                MultipartUpload={"Parts": [
                    {"PartNumber": p.number, "ETag": p.etag}
                    for p in ordered
                ]},
            )
            head = client.head_object(Bucket=self._bucket, Key=key)
            return int(head["ContentLength"])

        return await asyncio.to_thread(_do)

    async def abort_multipart(
        self, *, tmp_id: str, upload_id: str,
    ) -> None:
        client = self._client_for()
        key    = self._key(tmp_id)

        def _do() -> None:
            try:
                client.abort_multipart_upload(
                    Bucket=self._bucket, Key=key, UploadId=upload_id,
                )
            except client.exceptions.ClientError as e:
                # NoSuchUpload is fine — already cleaned up.
                code = e.response.get("Error", {}).get("Code", "")
                if code not in ("NoSuchUpload", "404"):
                    raise

        await asyncio.to_thread(_do)

    async def fetch(self, *, tmp_id: str, dest: Path) -> int:
        client = self._client_for()
        key    = self._key(tmp_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _do() -> int:
            # download_file uses range-GETs internally; for chapter
            # sizes (≤ a few hundred MB) the overhead is negligible
            # and we get retry-on-fault for free.
            client.download_file(self._bucket, key, str(dest))
            return dest.stat().st_size

        return await asyncio.to_thread(_do)

    async def delete(self, *, tmp_id: str) -> None:
        client = self._client_for()
        key    = self._key(tmp_id)

        def _do() -> None:
            try:
                client.delete_object(Bucket=self._bucket, Key=key)
            except client.exceptions.ClientError as e:
                # Already gone — fine.
                code = e.response.get("Error", {}).get("Code", "")
                if code not in ("NoSuchKey", "404"):
                    raise

        await asyncio.to_thread(_do)


class LocalInbox:
    """Filesystem inbox simulating multipart upload.

    Layout:
        <root>/<tmp_id>/upload_id      — random token
        <root>/<tmp_id>/parts/<n>      — raw bytes for part `n`
        <root>/<tmp_id>.zip            — assembled object after complete

    `create_multipart` returns `file://` URLs the dev API server itself
    accepts via a sibling local-PUT route — keeps the SDK code path
    identical between dev and prod. (Wired up in `routes/upload.py`.)
    """

    backend_name = "local"

    def __init__(self, *, root: Path, base_url: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # `base_url` lets us hand back URLs the SDK can PUT to without
        # caring whether the server is on localhost or a tailnet host.
        self._base = base_url.rstrip("/")

    def _dir(self, tmp_id: str) -> Path:
        return self._root / tmp_id

    def _zip(self, tmp_id: str) -> Path:
        return self._root / f"{tmp_id}.zip"

    async def create_multipart(
        self, *, tmp_id: str, part_count: int, part_size: int,
    ) -> tuple[str, list[PartUrl]]:
        d = self._dir(tmp_id)
        (d / "parts").mkdir(parents=True, exist_ok=True)
        upload_id = hashlib.sha256(tmp_id.encode()).hexdigest()[:16]
        (d / "upload_id").write_text(upload_id)
        urls = [
            PartUrl(
                number=n,
                url=f"{self._base}/api/_inbox/{tmp_id}/{upload_id}/{n}",
            )
            for n in range(1, part_count + 1)
        ]
        return upload_id, urls

    async def complete_multipart(
        self, *, tmp_id: str, upload_id: str, parts: list[CompletedPart],
    ) -> int:
        d = self._dir(tmp_id)
        stored = (d / "upload_id").read_text().strip()
        if stored != upload_id:
            raise ValueError("upload_id mismatch")

        ordered = sorted(parts, key=lambda p: p.number)
        out = self._zip(tmp_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fout:
            for p in ordered:
                src = d / "parts" / str(p.number)
                if not src.exists():
                    raise FileNotFoundError(f"missing part {p.number}")
                # ETag in local mode is sha256 of the part bytes.
                actual = _sha256_file(src)
                if actual != p.etag:
                    raise ValueError(
                        f"ETag mismatch on part {p.number}: "
                        f"expected {p.etag!r}, got {actual!r}",
                    )
                with src.open("rb") as fin:
                    shutil.copyfileobj(fin, fout)
        # The parts dir is no longer needed.
        shutil.rmtree(d, ignore_errors=True)
        return out.stat().st_size

    async def abort_multipart(
        self, *, tmp_id: str, upload_id: str,
    ) -> None:
        shutil.rmtree(self._dir(tmp_id), ignore_errors=True)

    async def fetch(self, *, tmp_id: str, dest: Path) -> int:
        src = self._zip(tmp_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest.stat().st_size

    async def delete(self, *, tmp_id: str) -> None:
        self._zip(tmp_id).unlink(missing_ok=True)
        shutil.rmtree(self._dir(tmp_id), ignore_errors=True)

    def part_path(self, tmp_id: str, upload_id: str, number: int) -> Path:
        d = self._dir(tmp_id)
        stored = (d / "upload_id").read_text().strip() if (d / "upload_id").exists() else None
        if stored != upload_id:
            raise ValueError("upload_id mismatch")
        return d / "parts" / str(number)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_inbox(cfg: StorageConfig, *, paths_root: Path, base_url: str) -> ChapterInbox:
    """Construct the configured inbox backend.

    `paths_root` is the dev artifact root (used by LocalInbox).
    `base_url` is the engine's public origin (used by LocalInbox to
    produce PUT URLs the SDK can hit).
    """
    spec = cfg.inbox
    if spec.type == "s3":
        if not (spec.s3_endpoint and spec.s3_access_key_id
                and spec.s3_secret_access_key and spec.s3_bucket):
            raise RuntimeError(
                "storage.inbox.type=s3 requires s3_endpoint, s3_bucket, "
                "s3_access_key_id, s3_secret_access_key.",
            )
        return S3Inbox(
            endpoint=spec.s3_endpoint,
            bucket=spec.s3_bucket,
            region=spec.s3_region or "auto",
            access_key=spec.s3_access_key_id,
            secret_key=spec.s3_secret_access_key,
            prefix=spec.s3_prefix or "tmp/",
        )
    if spec.type == "local":
        return LocalInbox(root=paths_root / "_inbox", base_url=base_url)
    raise RuntimeError(f"unknown storage.inbox.type: {spec.type!r}")
