"""Lens detector package — two-phase OCR (tile + bubble)."""

from .detector import LensBlocksDetector, LensUnavailableError


__all__ = ["LensBlocksDetector", "LensUnavailableError"]
