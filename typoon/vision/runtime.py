"""VisionRuntime — assembled stages from a VisionPipelineSpec.

Holds concrete instances (detector, grouper, recognizer, eraser) plus
per-stage semaphores for bounded concurrency. Built once per chapter.

Stateful by design: model instances cache loaded weights. Reuse across
pages within a chapter run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import TextDetector, TextEraser, TextGrouper, TextRecognizer
from .pipeline import (
    DetectorId,
    EraserId,
    GrouperId,
    RecognizerId,
    VisionPipelineSpec,
)


__all__ = ["VisionRuntime", "build_vision_runtime"]


@dataclass(slots=True)
class VisionRuntime:
    """Runtime instances + concurrency gates for one VisionPipelineSpec.

    Semaphores live here (not on individual stages) because they are
    pipeline-wide budgets shared across pages. Stages acquire by entering
    `runtime.detect_gate` / `runtime.erase_gate` context managers.
    """

    spec:       VisionPipelineSpec
    detector:   TextDetector
    grouper:    TextGrouper
    recognizer: TextRecognizer | None
    eraser:     TextEraser

    page_gate:   asyncio.Semaphore = field(init=False)
    detect_gate: asyncio.Semaphore = field(init=False)
    erase_gate:  asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.page_gate   = asyncio.Semaphore(self.spec.page_concurrency)
        self.detect_gate = asyncio.Semaphore(self.spec.detect_concurrency)
        self.erase_gate  = asyncio.Semaphore(self.spec.erase_concurrency)


def build_vision_runtime(
    spec: VisionPipelineSpec,
    *,
    models_dir: Path,
    source_lang: str | None = None,
    lens_endpoint: str | None = None,
) -> VisionRuntime:
    """Wire a spec into concrete stage instances.

    `source_lang` is a hint for backends that need a recogniser locale
    (e.g. Apple Vision). The actual `detect/group/recognize` calls receive
    the language at call-time so it can change per chapter.

    `lens_endpoint` overrides the default Lens upstream URL (Discord
    Activity proxy support). Bing has no equivalent override—its 302
    redirect is incompatible with CF Worker auto-follow.
    """
    detector   = _build_detector(
        spec.detector, models_dir, lens_endpoint=lens_endpoint,
    )
    grouper    = _build_grouper(spec.grouper, models_dir)
    recognizer = (
        _build_recognizer(spec.recognizer, source_lang)
        if spec.recognizer != "none"
        else None
    )
    eraser     = _build_eraser(spec.eraser, models_dir)
    return VisionRuntime(
        spec=spec,
        detector=detector,
        grouper=grouper,
        recognizer=recognizer,
        eraser=eraser,
    )


# ─── Per-stage factories ──────────────────────────────────────────────────


def _build_detector(
    kind: DetectorId,
    models_dir: Path,
    *,
    lens_endpoint: str | None,
) -> TextDetector:
    match kind:
        case "lens_blocks":
            from .detectors.lens_blocks import LensBlocksDetector
            return LensBlocksDetector(endpoint=lens_endpoint)
        case "bing_blocks":
            from .detectors.bing_blocks import BingBlocksDetector
            return BingBlocksDetector()
        case "ppocr_dbnet":
            from .detectors.ppocr_dbnet import PPOCRDetector
            return PPOCRDetector(
                model_path=models_dir / "ppocr-det.safetensors",
                config_path=models_dir / "ppocr-det-config.json",
            )


def _build_grouper(kind: GrouperId, models_dir: Path) -> TextGrouper:
    match kind:
        case "lens_native":
            from .groupers.lens_native import LensNativeGrouper
            return LensNativeGrouper()
        case "ppocr_yolo_union_find":
            from .groupers.ppocr_yolo_union_find import PPOCRYoloUnionFindGrouper
            return PPOCRYoloUnionFindGrouper(models_dir=models_dir)


def _build_recognizer(kind: RecognizerId, source_lang: str | None) -> TextRecognizer:
    match kind:
        case "manga_ocr":
            from .ocr import manga_ocr as _mo
            from .recognizers import MangaOcrRecognizer
            if not _mo.is_available():
                raise RuntimeError(
                    "manga-ocr backend requested but not installed; "
                    "install with `pip install transformers torch fugashi`"
                )
            return MangaOcrRecognizer(_mo.MangaOcrCropOcr())
        case "apple_vision":
            from .ocr import apple_vision as _av
            from .recognizers import PageOcrRecognizer
            if not _av.is_available():
                raise RuntimeError("apple_vision OCR not available on this host")
            return PageOcrRecognizer(_av.AppleVisionPageOcr(), name="apple_vision")
        case "windows_ocr":
            from .ocr import windows as _w
            from .recognizers import PageOcrRecognizer
            if not _w.is_available():
                raise RuntimeError("windows OCR not available on this host")
            return PageOcrRecognizer(_w.WindowsOcrPageOcr(), name="windows_ocr")
        case "tesseract":
            from .ocr import tesseract as _t
            from .recognizers import PageOcrRecognizer
            if not _t.is_available():
                raise RuntimeError("tesseract OCR not available on this host")
            return PageOcrRecognizer(_t.TesseractPageOcr(), name="tesseract")
        case "none":
            raise ValueError("recognizer=none should be handled by caller")


def _build_eraser(kind: EraserId, models_dir: Path) -> TextEraser:
    match kind:
        case "aot_gan":
            from .erasers import AOTGANEraser
            return AOTGANEraser(models_dir=models_dir)
        case "median_only":
            from .erasers import MedianEraser
            return MedianEraser()
