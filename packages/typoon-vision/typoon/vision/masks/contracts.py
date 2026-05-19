"""Mask strategy contracts — pure domain, no CV deps."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from typoon.vision.contracts import BubbleGroup, TextBlock, TextMask


__all__ = ["MaskStrategyId", "MaskRecipe", "MaskStrategyImpl"]


class MaskStrategyId(str, Enum):
    PIXEL_SEG    = "pixel_seg"    # threshold word pixels + morphological close
    OBB_PER_LINE = "obb_per_line" # oriented bbox per line (burst/SFX)
    RECT_DILATE  = "rect_dilate"  # word union rect + dilate (fallback, no image)
    CTD_UNET     = "ctd_unet"     # CTD UNet seg-only, clipped to group bbox


@dataclass(frozen=True, slots=True)
class MaskRecipe:
    """What strategy to use — no pixel data.

    Serialisable, tiny, safe to store alongside BubbleGroup.
    Actual pixels are produced by materialize().
    """
    strategy: MaskStrategyId
    shape_kind: str = "dialogue"


@runtime_checkable
class MaskStrategyImpl(Protocol):
    name: str

    def build(
        self,
        group: "BubbleGroup",
        members: "tuple[TextBlock, ...]",
        image: "np.ndarray | None",
    ) -> "tuple[TextMask, ...]": ...
