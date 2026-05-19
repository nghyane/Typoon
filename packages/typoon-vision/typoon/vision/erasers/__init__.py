"""Eraser backends — remove text from a page canvas.

Public surface:
  HybridEraser  — TeLeA for uniform backgrounds, AOT-GAN for complex.
                  This is the only eraser wired into the prod pipeline.

  Backends (InpaintBackend protocol):
    TeLeABackend          — cv2.INPAINT_TELEA, no model, ~90ms/page
    AOTGANBackend         — local AOT-GAN ONNX/CoreML inpainter
    RemoteInpaintBackend  — base class for HTTP-backed services
    CfSd15InpaintBackend  — Cloudflare Workers AI SD1.5 inpaint
    Flux2KleinInpaintBackend — FLUX2 Klein inpaint

  Routing helpers (driving HybridEraser):
    classify_masks   — luminance-spread split into uniform / complex
    build_page_mask  — OR per-mask TextMasks into a single page-level mask
    inpaint_region   — call a backend with the page mask, paste back
"""

from __future__ import annotations

import logging

from .hybrid import HybridEraser
from .routing import (
    classify_masks,
    build_page_mask,
    inpaint_region,
    _is_uniform_background,
)
from .backends import (
    InpaintBackend,
    TeLeABackend,
    AOTGANBackend,
    RemoteInpaintBackend,
    CfSd15InpaintBackend,
    Flux2KleinInpaintBackend,
)


__all__ = [
    "HybridEraser",
    "InpaintBackend",
    "TeLeABackend",
    "AOTGANBackend",
    "RemoteInpaintBackend",
    "CfSd15InpaintBackend",
    "Flux2KleinInpaintBackend",
    "classify_masks",
    "build_page_mask",
    "inpaint_region",
    "_is_uniform_background",
]

logger = logging.getLogger(__name__)
