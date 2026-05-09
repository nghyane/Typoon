"""Vision model runtime adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from typoon.models import ModelHub
from typoon.vision.erase import Eraser


class VisionRuntime:
    """Owns vision model instances and page-local scan primitives."""

    def __init__(
        self,
        scanner,
        eraser: Eraser,
        hub: ModelHub,
        *,
        bubble_scope_imgsz: int = 640,
    ) -> None:
        self.scanner = scanner
        self.eraser = eraser
        self._hub = hub
        self._bubble_scope_imgsz = bubble_scope_imgsz
        self._yolo_model = None

    @staticmethod
    def from_config(config=None, paths=None):
        from typoon.config import load_config
        from typoon.vision.scanner import create_scanner

        if config is None or paths is None:
            config, paths = load_config()
        hub = ModelHub(Path(config.models_dir))
        runtime = VisionRuntime(
            scanner=create_scanner(hub=hub),
            eraser=Eraser(str(hub.dir)),
            hub=hub,
            bubble_scope_imgsz=config.bubble_scope_imgsz,
        )
        return runtime, config, paths

    def _get_yolo_model(self) -> Any | None:
        if self._yolo_model is None:
            from typoon.vision.bubble_scope import load_yolo_model
            import sys
            if sys.platform == "darwin":
                path = self._hub.resolve("bubble-scope-yolov8m.mlpackage")
            else:
                path = self._hub.resolve("bubble-scope-yolov8m.pt")
            self._yolo_model = load_yolo_model(path)
        return self._yolo_model

    def scan_page_state(self, image: np.ndarray, *, source_lang: str | None = None):
        """Run full vision pipeline and return ScanState for artifact writing.

        `source_lang` selects the OCR recognizer language (project's
        ISO 639-1 code, e.g. "ja", "ko", "zh"). When omitted, the
        backend falls back to English — pass it explicitly for any
        non-English project or detected text gets dropped as noise.
        """
        from typoon.vision.grouping import scan_page
        self.scanner.set_language(source_lang)
        return scan_page(
            self.scanner,
            image,
            yolo_model=self._get_yolo_model(),
            yolo_imgsz=self._bubble_scope_imgsz,
        )
