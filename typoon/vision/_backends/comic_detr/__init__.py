"""Comic text + bubble detector backend.

Wraps the ONNX RT-DETRv2-small int8 model from
``ogkalu/comic-text-and-bubble-detector`` (10.6MB). Detects three
classes on a comic / manga / webtoon page:

  * ``bubble`` (0)      — speech-bubble outline rectangle
  * ``text_bubble`` (1) — text region inside a bubble
  * ``text_free`` (2)   — SFX / narration outside any bubble

Used by the ``lens`` preset to (a) provide bubble outlines for render
fit expansion and (b) anchor Lens-emitted text blocks into bubble
groups via spatial join, replacing the heuristic tategaki chain
clustering.

Model is loaded lazily on first ``detect()`` call. Sessions are
process-wide singletons keyed by model path.

Output coord convention: ``orig_target_sizes`` is ``[width, height]``
for this model (verified against the page extent on probe fixtures).
The HuggingFace RT-DETR convention is ``[height, width]`` so this is
an export quirk specific to ogkalu's repo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


__all__ = ["ComicDet", "ComicDetrSession", "load_session"]

logger = logging.getLogger(__name__)


# ─── Output type ──────────────────────────────────────────────────────────


ComicClass = Literal["bubble", "text_bubble", "text_free"]


@dataclass(frozen=True, slots=True)
class ComicDet:
    """One detection from the comic_detr model in page-pixel coords."""
    cls:  ComicClass
    conf: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) page pixels


# ─── Constants ────────────────────────────────────────────────────────────


_INPUT_SIZE      = 640
_DEFAULT_CONF    = 0.30
_BBOX_PAD_PX     = 6     # expand bboxes by this many px on each side

_CLASS_NAMES: dict[int, ComicClass] = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}


# ─── Session ──────────────────────────────────────────────────────────────


_SESSION_CACHE: dict[str, "ComicDetrSession"] = {}


class ComicDetrSession:
    """ONNX runtime session wrapper. Thread-safe per session instance.

    ``infer()`` does the resize + transpose + scale; callers don't
    touch tensor layout. CoreML execution provider preferred on macOS,
    CPU fallback elsewhere.
    """

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        providers: list[str] = []
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._model_path = model_path
        logger.info(
            "comic_detr session built: %s providers=%s",
            Path(model_path).name, providers,
        )

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = _DEFAULT_CONF,
    ) -> list[ComicDet]:
        """Run inference on a page; return all detections above threshold."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {image.shape}")
        h, w = image.shape[:2]
        resized = cv2.resize(
            image, (_INPUT_SIZE, _INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        arr = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        # Axis order: this model takes [W, H], not the HuggingFace [H, W].
        sizes = np.array([[w, h]], dtype=np.int64)
        labels, boxes, scores = self._sess.run(
            None, {"images": arr, "orig_target_sizes": sizes},
        )
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        out: list[ComicDet] = []
        for i in np.where(scores >= conf_threshold)[0]:
            cls_id = int(labels[i])
            if cls_id not in _CLASS_NAMES:
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            x1 = max(0, int(x1) - _BBOX_PAD_PX)
            y1 = max(0, int(y1) - _BBOX_PAD_PX)
            x2 = min(w, int(x2) + _BBOX_PAD_PX)
            y2 = min(h, int(y2) + _BBOX_PAD_PX)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(ComicDet(
                cls=_CLASS_NAMES[cls_id],
                conf=float(scores[i]),
                bbox=(x1, y1, x2, y2),
            ))
        return out


def load_session(model_path: str) -> ComicDetrSession:
    """Return a cached session for the given model path."""
    sess = _SESSION_CACHE.get(model_path)
    if sess is None:
        sess = ComicDetrSession(model_path)
        _SESSION_CACHE[model_path] = sess
    return sess
