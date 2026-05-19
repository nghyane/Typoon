"""Centralized model resolver — local dir first, HuggingFace Hub fallback.

.mlpackage folders are downloaded via snapshot_download and copied with symlink
resolution because CoreML needs real files.

Usage:
    hub = ModelHub(Path("models"))
    det_path = hub.resolve("ppocr-det.safetensors")
    ml_path  = hub.resolve("bubble-scope-yolov8m.mlpackage")
    ctd_path = hub.resolve_ctd("ctd-yolo-v5.safetensors")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO     = "nghyane/typoon"
HF_CTD_REPO = "mayocream/comic-text-detector"

# All model assets for the main pipeline. Folders end with /.
MANIFEST = [
    "ppocr-det.safetensors",
    "ppocr-det-config.json",
    "ppocr-det-preprocessor.json",
    "ppocr-det.onnx",
    "ppocr-det.onnx.data",
    "ppocr-det.mlpackage/",
    "bubble-scope-yolov8m.pt",
    "bubble-scope-yolov8m.mlpackage/",
]

# CTD weights live in mayocream/comic-text-detector.
# Local copies use ctd-* prefix to avoid name collisions.
CTD_MANIFEST: dict[str, str] = {
    "ctd-yolo-v5.safetensors": "yolo-v5.safetensors",
    "ctd-unet.safetensors":    "unet.safetensors",
    "ctd-dbnet.safetensors":   "dbnet.safetensors",
}

CTD_ONNX_REPO  = "mayocream/comic-text-detector-onnx"
CTD_ONNX_FILE  = "comic-text-detector.onnx"
CTD_ONNX_LOCAL = "ctd.onnx"

# Manga / comic text + bubble detector. RT-DETRv2 small int8 ONNX, 10.6MB.
# Fine-tuned on 11k manga / webtoon / manhua images. Three classes:
# bubble (0), text_bubble (1), text_free (2). Used by the lens preset to
# (a) recover Lens-missed text regions and (b) provide bubble outlines
# for render fit. License: Apache-2.0.
COMIC_DETR_REPO  = "ogkalu/comic-text-and-bubble-detector"
COMIC_DETR_FILE  = "detector-v4-s_int8.onnx"
COMIC_DETR_LOCAL = "comic-detr-v4s-int8.onnx"


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

    def resolve_ctd_onnx(self) -> str:
        """Resolve the CTD ONNX model (mayocream/comic-text-detector-onnx)."""
        local = self._dir / CTD_ONNX_LOCAL
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s...", CTD_ONNX_FILE, CTD_ONNX_REPO)
        src = Path(hf_hub_download(repo_id=CTD_ONNX_REPO, filename=CTD_ONNX_FILE))
        self._dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, local)
        logger.info("Saved CTD ONNX -> %s", local)
        return str(local)

    def resolve_comic_detr(self) -> str:
        """Resolve the comic-text-and-bubble-detector ONNX (ogkalu/...)."""
        local = self._dir / COMIC_DETR_LOCAL
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s...", COMIC_DETR_FILE, COMIC_DETR_REPO)
        src = Path(hf_hub_download(repo_id=COMIC_DETR_REPO, filename=COMIC_DETR_FILE))
        self._dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)
        logger.info("Saved comic-detr ONNX -> %s", local)
        return str(local)

    def resolve_ctd(self, local_name: str) -> str:
        """Resolve a CTD model file (mayocream/comic-text-detector).

        `local_name` is the local alias (e.g. 'ctd-yolo-v5.safetensors').
        Downloads from HF_CTD_REPO on first use, then caches locally.
        """
        local = self._dir / local_name
        if local.exists():
            return str(local)
        hf_name = CTD_MANIFEST.get(local_name)
        if hf_name is None:
            raise KeyError(f"Unknown CTD model: {local_name!r}")
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s...", hf_name, HF_CTD_REPO)
        src = Path(hf_hub_download(repo_id=HF_CTD_REPO, filename=hf_name))
        self._dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)
        logger.info("Saved CTD model -> %s", local)
        return str(local)

    def ensure_all(self) -> None:
        """Download all main pipeline models if missing."""
        for name in MANIFEST:
            self.resolve(name)

    def ensure_ctd(self) -> None:
        """Download all CTD model weights if missing."""
        for local_name in CTD_MANIFEST:
            self.resolve_ctd(local_name)

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
