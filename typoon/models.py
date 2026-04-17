"""Model resolver — local dir first, HuggingFace Hub fallback.

Usage:
    models = ModelHub(models_dir="models")
    det_path = models.resolve("ppocr-det.safetensors")
    det_config = models.resolve("ppocr-det-config.json")
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO = "nghyane/typoon"

# v2 model files
REQUIRED = [
    "ppocr-det.safetensors",
    "ppocr-det-config.json",
    "ppocr-det.onnx",
    "aot-inpaint.mlpackage",
    "aot-inpaint.onnx",
]


class ModelHub:
    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    def resolve(self, filename: str) -> str:
        """Resolve model file: local first, HF Hub download fallback."""
        local = self._dir / filename
        if local.exists():
            return str(local)

        logger.info("Downloading %s from %s...", filename, HF_REPO)
        path = _download(filename)
        logger.info("Cached: %s", path)
        return str(path)

    def ensure_all(self) -> None:
        """Download all required models if missing."""
        for f in REQUIRED:
            self.resolve(f)


def _download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO, filename=filename))
