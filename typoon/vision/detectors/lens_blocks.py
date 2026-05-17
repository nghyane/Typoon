"""Backwards-compat re-export — the real implementation lives in
``typoon.vision.detectors.lens``.

Kept so external probe scripts that ``from typoon.vision.detectors.lens_blocks
import LensBlocksDetector`` still work without churn.
"""

from .lens import LensBlocksDetector, LensUnavailableError


__all__ = ["LensBlocksDetector", "LensUnavailableError"]
