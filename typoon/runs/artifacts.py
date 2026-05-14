"""Filesystem artifacts for repeatable visual runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


STAGE_DIRS = (
    "01_prepare",
    "02_detect",
    "03_group",
    "04_ocr",
    "05_brief",
    "06_translate",
    "07_render",
    "final",
)


class ArtifactSink(Protocol):
    root: Path

    def write_json(self, stage: str, name: str, data: Any) -> Path: ...
    def write_image(self, stage: str, name: str, image: np.ndarray) -> Path: ...
    def write_bytes(self, stage: str, name: str, data: bytes) -> Path: ...


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    stages: tuple[str, ...] = STAGE_DIRS

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "run_id": self.run_id, "stages": list(self.stages)}


class FileArtifactSink:
    """Writes run artifacts under debug-runs/<run-id>/ by stage."""

    def __init__(self, root: Path, run_id: str, *, clean: bool = True) -> None:
        self.root = Path(root) / run_id
        self.run_id = run_id
        if clean and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGE_DIRS:
            (self.root / stage).mkdir(parents=True, exist_ok=True)
        self.write_json(".", "manifest.json", RunManifest(run_id).to_dict())

    def write_json(self, stage: str, name: str, data: Any) -> Path:
        out = self._path(stage, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        return out

    def write_image(self, stage: str, name: str, image: np.ndarray) -> Path:
        out = self._path(stage, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        if not cv2.imwrite(str(out), bgr):
            raise RuntimeError(f"Failed to write image: {out}")
        return out

    def write_bytes(self, stage: str, name: str, data: bytes) -> Path:
        out = self._path(stage, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def _path(self, stage: str, name: str) -> Path:
        if stage in ("", "."):
            return self.root / name
        return self.root / stage / name
