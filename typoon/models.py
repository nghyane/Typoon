"""Centralized model resolver — local dir first, HuggingFace Hub fallback.

.mlpackage folders are downloaded via snapshot_download and copied with symlink
resolution because CoreML needs real files.

Usage:
    hub = ModelHub(Path("models"))
    det_path = hub.resolve("ppocr-det.safetensors")
    ml_path  = hub.resolve("bubble-scope-yolov8m.mlpackage")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO = "nghyane/typoon"

# All model assets. Folders end with /.
MANIFEST = [
    "ppocr-det.safetensors",
    "ppocr-det-config.json",
    "ppocr-det-preprocessor.json",
    "ppocr-det.onnx",
    "ppocr-det.onnx.data",
    "ppocr-det.mlpackage/",
    "aot-inpaint.onnx",
    "aot-inpaint.mlpackage/",
    "bubble-scope-yolov8m.pt",
    "bubble-scope-yolov8m.mlpackage/",
]


class ModelHub:
    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    @property
    def dir(self) -> Path:
        return self._dir

    def resolve(self, name: str) -> str:
        """Resolve model file or folder: local first, HF download fallback.

        For .mlpackage folders, pass name with or without trailing slash.
        Returns absolute path string.
        """
        clean = name.rstrip("/")
        local = self._dir / clean
        if local.exists():
            return str(local)

        if clean.endswith(".mlpackage"):
            path = self._download_folder(clean)
        else:
            path = self._download_file(clean)
        logger.info("Resolved %s -> %s", name, path)
        return str(path)

    def _download_file(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s from %s...", filename, HF_REPO)
        return Path(hf_hub_download(repo_id=HF_REPO, filename=filename))

    def _download_folder(self, folder_name: str) -> Path:
        """Download .mlpackage folder and copy to models_dir (resolves symlinks)."""
        dest = self._dir / folder_name
        if dest.exists():
            return dest

        from huggingface_hub import snapshot_download

        logger.info("Downloading %s/ from %s...", folder_name, HF_REPO)
        prefix = f"{folder_name}/"
        snap = Path(snapshot_download(HF_REPO, allow_patterns=[f"{prefix}*"]))
        src = snap / folder_name
        if not src.exists():
            raise FileNotFoundError(f"{folder_name} not found in {HF_REPO}")

        self._dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, copy_function=shutil.copy2)
        return dest

    def ensure_all(self) -> None:
        """Download all required models if missing."""
        for name in MANIFEST:
            self.resolve(name)
