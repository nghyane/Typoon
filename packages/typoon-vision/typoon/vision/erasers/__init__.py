"""Eraser public API.

Production entry point:
  TextEraser  — route per-mask to uniform or complex PageInpainter.

Page-level inpaint driver:
  FullPageInpainter  — single backend call on the full page.
  PageInpainter      — Protocol.

Backends (InpaintBackend Protocol):
  TeLeABackend              — cv2 TELEA, no model, ~90ms/page
  RemoteInpaintBackend      — base for HTTP-backed services
  CfSd15InpaintBackend      — Cloudflare Workers AI SD1.5
  Flux2KleinInpaintBackend  — FLUX2 Klein

Routing helpers:
  partition_by_background  — split masks by luminance spread
  build_page_mask          — OR per-mask images into a page binary mask
"""

from __future__ import annotations

import logging

from .eraser import TextEraser
from .inpaint import FullPageInpainter, PageInpainter
from .routing import build_page_mask, partition_by_background
from .backends import (
    InpaintBackend,
    RemoteInpaintBackend,
    CfSd15InpaintBackend,
    Flux2KleinInpaintBackend,
    TeLeABackend,
)

__all__ = [
    # Eraser
    "TextEraser",
    # Page inpaint drivers
    "PageInpainter",
    "FullPageInpainter",
    # Backends
    "InpaintBackend",
    "TeLeABackend",
    "RemoteInpaintBackend",
    "CfSd15InpaintBackend",
    "Flux2KleinInpaintBackend",
    # Routing helpers
    "partition_by_background",
    "build_page_mask",
]

logger = logging.getLogger(__name__)
