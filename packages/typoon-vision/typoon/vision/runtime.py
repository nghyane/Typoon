"""VisionRuntime — assembled stages from a VisionPipelineSpec."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import TextDetector, TextEraser, TextGrouper, TextRecognizer, Eraser
from .masks.ctd_seg_runner import CtdSegRunner
from .pipeline import DetectorId, EraserId, GrouperId, RecognizerId, VisionPipelineSpec


__all__ = ["VisionRuntime", "build_vision_runtime"]


@dataclass(slots=True)
class VisionRuntime:
    """Runtime instances + concurrency gates for one VisionPipelineSpec."""

    spec:       VisionPipelineSpec
    detector:   TextDetector
    grouper:    TextGrouper
    recognizer: TextRecognizer | None
    eraser:     TextEraser
    ctd_seg:    CtdSegRunner | None = None  # seg-only UNet, optional

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
    """Wire a spec into concrete stage instances."""
    detector   = _build_detector(spec.detector, models_dir, lens_endpoint=lens_endpoint)
    grouper    = _build_grouper(spec.grouper)
    recognizer = _build_recognizer(spec.recognizer, source_lang) if spec.recognizer != "none" else None
    eraser     = _build_eraser(spec.eraser, models_dir)
    ctd_seg    = _build_ctd_seg(models_dir, source_lang)
    return VisionRuntime(
        spec=spec,
        detector=detector,
        grouper=grouper,
        recognizer=recognizer,
        eraser=eraser,
        ctd_seg=ctd_seg,
    )


def _build_detector(
    kind: DetectorId,
    models_dir: Path,
    *,
    lens_endpoint: str | None,
) -> TextDetector:
    match kind:
        case "lens_blocks":
            from .detectors.lens import LensBlocksDetector
            from ._backends.comic_detr import load_session
            from typoon.models import ModelHub
            comic = load_session(ModelHub(models_dir).resolve_comic_detr())
            return LensBlocksDetector(endpoint=lens_endpoint, comic_detr=comic)
        case "ctd_blocks":
            from .detectors.ctd_blocks import CTDDetector
            from typoon.models import ModelHub
            return CTDDetector(onnx_path=ModelHub(models_dir).resolve_ctd_onnx())


def _build_grouper(kind: GrouperId) -> TextGrouper:
    match kind:
        case "lens_native":
            from .groupers.lens_native import LensNativeGrouper
            return LensNativeGrouper()
        case "ctd_native":
            from .groupers.ctd_native import CTDNativeGrouper
            return CTDNativeGrouper()


def _build_recognizer(kind: RecognizerId, source_lang: str | None) -> TextRecognizer:
    match kind:
        case "manga_ocr":
            from .ocr import manga_ocr as _mo
            from .recognizers import MangaOcrRecognizer
            if not _mo.is_available():
                raise RuntimeError("manga-ocr not installed")
            return MangaOcrRecognizer(_mo.MangaOcrCropOcr())
        case "apple_vision":
            from .ocr import apple_vision as _av
            from .recognizers import PageOcrRecognizer
            if not _av.is_available():
                raise RuntimeError("apple_vision OCR not available")
            return PageOcrRecognizer(_av.AppleVisionPageOcr(), name="apple_vision")
        case "windows_ocr":
            from .ocr import windows as _w
            from .recognizers import PageOcrRecognizer
            if not _w.is_available():
                raise RuntimeError("windows OCR not available")
            return PageOcrRecognizer(_w.WindowsOcrPageOcr(), name="windows_ocr")
        case "tesseract":
            from .ocr import tesseract as _t
            from .recognizers import PageOcrRecognizer
            if not _t.is_available():
                raise RuntimeError("tesseract not available")
            return PageOcrRecognizer(_t.TesseractPageOcr(), name="tesseract")
        case "none":
            raise ValueError("recognizer=none should be handled by caller")


def _build_ctd_seg(models_dir: Path, source_lang: str | None) -> CtdSegRunner | None:
    """Load CtdSegRunner only for Japanese source — CTD trained on ja manga."""
    if source_lang and source_lang.lower().startswith("ja"):
        onnx = models_dir / "ctd.onnx"
        if onnx.exists():
            return CtdSegRunner(onnx)
    return None


def _build_eraser(kind: EraserId, models_dir: Path) -> Eraser:
    match kind:
        case "text":
            from .erasers import TextEraser, FullPageInpainter, TiledInpainter
            from .erasers.inpaint import AreaGatedInpainter
            from .erasers.backends import AOTGANBackend, TeLeABackend
            return TextEraser(
                uniform_inpainter=FullPageInpainter(TeLeABackend()),
                complex_inpainter=AreaGatedInpainter(
                    small_inpainter=FullPageInpainter(TeLeABackend()),
                    large_inpainter=TiledInpainter(
                        AOTGANBackend(models_dir),
                        context_px=64,
                    ),
                    area_threshold=1000,
                ),
            )
