"""Eraser public API.

Production entry point:
  TextEraser  — route per-mask to uniform or complex PageInpainter.

Page-level inpaint drivers:
  FullPageInpainter  — single backend call on the full page.
  TiledInpainter     — per-blob crop, backend call, paste back.
  PageInpainter      — Protocol.

Backends (InpaintBackend Protocol):
  TeLeABackend          — cv2 TELEA, no model, ~90ms/page
  RemoteInpaintBackend  — base for HTTP-backed services
  TyphoonInpaintBackend — Rust/Candle inpaint container (spike/inpaint)

Routing helpers:
  partition_by_background  — split masks by luminance spread
  build_page_mask          — OR per-mask images into a page binary mask
"""

from __future__ import annotations

import logging

from .eraser import TextEraser
from .inpaint import FullPageInpainter, PageInpainter, TiledInpainter
from .routing import build_page_mask, partition_by_background
from .backends import (
    InpaintBackend,
    RemoteInpaintBackend,
    TyphoonInpaintBackend,
    TeLeABackend,
)

__all__ = [
    "TextEraser",
    "PageInpainter",
    "FullPageInpainter",
    "TiledInpainter",
    "InpaintBackend",
    "TeLeABackend",
    "RemoteInpaintBackend",
    "TyphoonInpaintBackend",
    "partition_by_background",
    "build_page_mask",
]

logger = logging.getLogger(__name__)
