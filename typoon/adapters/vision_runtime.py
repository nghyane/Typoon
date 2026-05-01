"""Vision model runtime adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from typoon.models import ModelHub
from typoon.runs.events import Hook, ModelsUnloaded
from typoon.vision.erase import Eraser
from typoon.vision.types import DetectedGroup

_NO_HOOK = Hook()


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

    def unload_scan_models(self, hook: Hook = _NO_HOOK) -> None:
        self.scanner = None  # type: ignore[assignment]
        hook.on(ModelsUnloaded(stage="scan"))

    def unload_erase_models(self, hook: Hook = _NO_HOOK) -> None:
        self.eraser = None  # type: ignore[assignment]
        hook.on(ModelsUnloaded(stage="erase"))

    def ensure_scan_models(self) -> None:
        if self.scanner is not None:
            return
        from typoon.vision.scanner import create_scanner
        self.scanner = create_scanner(hub=self._hub)

    def ensure_erase_models(self) -> None:
        if self.eraser is not None:
            return
        self.eraser = Eraser(str(self._hub.dir))

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

    def scan_page_state(self, image: np.ndarray):
        """Run full vision pipeline and return ScanState for artifact writing."""
        from typoon.vision.grouping import scan_page
        return scan_page(
            self.scanner,
            image,
            yolo_model=self._get_yolo_model(),
            yolo_imgsz=self._bubble_scope_imgsz,
        )

    def scan_page(self, image: np.ndarray) -> list[DetectedGroup]:
        from typoon.vision.grouping import export_groups
        return export_groups(self.scan_page_state(image))
