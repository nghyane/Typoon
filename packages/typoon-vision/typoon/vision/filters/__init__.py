"""vision/filters public API."""

from __future__ import annotations

from typoon.vision.filters.pipeline import GroupFilter
from typoon.vision.filters.scorers.geometry import GeometryScorer
from typoon.vision.filters.scorers.oversize_sfx import OversizeSfxScorer
from typoon.vision.filters.scorers.rotation import RotationScorer
from typoon.vision.filters.scorers.watermark import WatermarkScorer


__all__ = [
    "GroupFilter",
    "GeometryScorer",
    "OversizeSfxScorer",
    "RotationScorer",
    "WatermarkScorer",
    "default_filter",
]


def default_filter() -> GroupFilter:
    """Standard filter for the lens preset."""
    return GroupFilter(
        scorers=(
            WatermarkScorer(),
            GeometryScorer(),
            RotationScorer(),
            OversizeSfxScorer(),
        ),
        threshold=1.0,
    )
