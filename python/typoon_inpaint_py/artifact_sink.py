"""ArtifactSink — write debug-runs artifacts from Python side."""
from __future__ import annotations

from pathlib import Path


class ArtifactSink:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def subdir(self, name: str) -> "ArtifactSink":
        return ArtifactSink(self._path / name)

    def write(self, name: str, data: bytes | str) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        p = self._path / name
        if isinstance(data, str):
            p.write_text(data, encoding="utf-8")
        else:
            p.write_bytes(data)


class NullSink(ArtifactSink):
    """No-op sink for production container."""

    def __init__(self) -> None:
        super().__init__(Path("/dev/null"))

    def subdir(self, name: str) -> "NullSink":
        return self

    def write(self, name: str, data: bytes | str) -> None:
        pass
