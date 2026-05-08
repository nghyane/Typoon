"""StorageRegistry — pipeline + public stores for the running process.

Two roles, one registry:

  pipeline (BlobStore)        prepared.bnl, masks.npz; cross-worker.
                              Single-host: LocalBlobStore.
                              Multi-host: HttpBlobStore over tailnet.

  public (ArtifactStore)      render.bnl; browser-facing.
                              Single-host: LocalArtifactStore.
                              Multi-host: HuggingFaceArtifactStore.

Plus a `readers` map from `archive_backend` → ArtifactStore so chapter
URL build can dispatch through the backend that actually wrote the
archive — chapters rendered against an old public store keep serving
after the operator switches.

Built once per process from `Config.storage`. Workers share the
pipeline; API hosts share the public reader. Lifetime ends with
`aclose()` (closes pooled HTTP clients).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typoon.adapters.artifact_store import (
    ArtifactStore, HuggingFaceArtifactStore, LocalArtifactStore,
)
from typoon.adapters.blob_store import BlobStore, LocalBlobStore
from typoon.adapters.http_blob_store import HttpBlobStore
from typoon.config import Config
from typoon.paths import Paths


@dataclass(frozen=True)
class StorageRegistry:
    pipeline: BlobStore
    public:   ArtifactStore
    readers:  dict[str, ArtifactStore] = field(default_factory=dict)

    def reader(self, backend: str) -> ArtifactStore:
        try:
            return self.readers[backend]
        except KeyError as e:
            raise RuntimeError(
                f"No artifact store configured for backend {backend!r}. "
                f"Available: {sorted(self.readers)}",
            ) from e

    async def aclose(self) -> None:
        await self.pipeline.aclose()
        if self.public is not self.pipeline:
            await self.public.aclose()


def build_storage(cfg: Config, paths: Paths) -> StorageRegistry:
    """Construct pipeline + public stores from config.

    Raises if a backend's required credentials are missing. The
    `readers` map carries one entry per configured public backend;
    chapters whose `archive_backend` no longer matches a configured
    reader will fail loud at URL build time so the operator notices
    and migrates / wipes them.
    """
    public = _build_public(cfg, paths)
    pipeline = _build_pipeline(cfg, paths)
    readers: dict[str, ArtifactStore] = {public.backend_name: public}
    return StorageRegistry(pipeline=pipeline, public=public, readers=readers)


def _build_public(cfg: Config, paths: Paths) -> ArtifactStore:
    spec = cfg.storage.public
    if spec.type == "huggingface":
        if not spec.hf_token:
            raise RuntimeError("storage.public.type=huggingface requires HF_TOKEN")
        return HuggingFaceArtifactStore(
            repo=spec.hf_repo,
            token=spec.hf_token,
            cdn_prefix=spec.cdn_prefix,
        )
    if spec.type == "local":
        return LocalArtifactStore(paths.artifacts)
    raise RuntimeError(f"unknown storage.public.type: {spec.type!r}")


def _build_pipeline(cfg: Config, paths: Paths) -> BlobStore:
    spec = cfg.storage.pipeline
    if spec.type == "http":
        if not spec.http_base_url or not spec.http_api_token:
            raise RuntimeError(
                "storage.pipeline.type=http requires http_base_url + "
                "http_api_token (TYPOON_PIPELINE_BASE_URL + "
                "TYPOON_PIPELINE_TOKEN)",
            )
        return HttpBlobStore(
            base_url=spec.http_base_url,
            api_token=spec.http_api_token,
        )
    if spec.type == "local":
        return LocalBlobStore(paths.artifacts)
    raise RuntimeError(f"unknown storage.pipeline.type: {spec.type!r}")
