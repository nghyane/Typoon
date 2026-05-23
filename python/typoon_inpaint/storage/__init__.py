"""Storage Protocol — abstract over LocalFs / R2 / Tigris FUSE."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Bytes in, bytes out.  Keys are POSIX strings ('prepared/12/0000.jpg')."""

    async def get(self, key: str) -> bytes: ...
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class LocalFsStorage:
    """Maps keys to `<root>/<key>` on local filesystem.

    Local workspace must mirror CF R2 layout:
        workspace/prepared/12/0000.jpg
        workspace/scan/12/0000.msgpack
        workspace/inpaint/12/0000.png
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        return self.root / key

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)


class R2Storage:
    """boto3 S3 → Cloudflare R2."""

    def __init__(
        self,
        bucket:     str,
        account_id: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        import boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._bucket = bucket

    @classmethod
    def from_env(cls) -> "R2Storage":
        return cls(
            bucket=os.environ["R2_BUCKET_NAME"],
            account_id=os.environ["R2_ACCOUNT_ID"],
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(
            lambda: self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        )

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        await asyncio.to_thread(
            lambda: self._s3.put_object(
                Bucket=self._bucket, Key=key,
                Body=data, ContentType=content_type,
            )
        )

    async def exists(self, key: str) -> bool:
        import botocore.exceptions
        try:
            await asyncio.to_thread(
                lambda: self._s3.head_object(Bucket=self._bucket, Key=key)
            )
            return True
        except botocore.exceptions.ClientError:
            return False
