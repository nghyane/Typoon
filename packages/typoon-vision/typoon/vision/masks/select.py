"""Strategy selector + materialize() — dispatch table, no if/else chains."""

from __future__ import annotations

import numpy as np

from typoon.vision.contracts import BubbleGroup, TextBlock, TextMask
from typoon.vision.masks.contracts import MaskRecipe, MaskStrategyId, MaskStrategyImpl
from typoon.vision.masks.ctd_unet import CtdUNetStrategy
from typoon.vision.masks.obb_per_line import ObbPerLineStrategy
from typoon.vision.masks.pixel_seg import PixelSegStrategy, RectDilateStrategy


__all__ = ["select_strategy", "materialize"]

_PIXEL_SEG    = PixelSegStrategy()
_OBB_PER_LINE = ObbPerLineStrategy()
_RECT_DILATE  = RectDilateStrategy()
_CTD_UNET     = CtdUNetStrategy()


def select_strategy(
    group: BubbleGroup,
    image: np.ndarray | None,
    bubble_mask: np.ndarray | None = None,
) -> MaskRecipe:
    """Pure function: (group, context) → MaskRecipe.

    Selection rules (priority order):
      burst/SFX          → OBB per-line (tight, no cross-group bleed)
      dialogue + CTD UNet available → CTD_UNET (best boundary quality)
      dialogue + image   → pixel_seg (word pixels + morphological close)
      dialogue + no image → rect_dilate fallback
    """
    if group.shape_kind == "burst":
        return MaskRecipe(strategy=MaskStrategyId.OBB_PER_LINE, shape_kind="burst")
    if bubble_mask is not None:
        return MaskRecipe(strategy=MaskStrategyId.CTD_UNET, shape_kind="dialogue")
    if image is not None:
        return MaskRecipe(strategy=MaskStrategyId.PIXEL_SEG, shape_kind="dialogue")
    return MaskRecipe(strategy=MaskStrategyId.RECT_DILATE, shape_kind="dialogue")


def materialize(
    recipe: MaskRecipe,
    group: BubbleGroup,
    members: tuple[TextBlock, ...],
    image: np.ndarray | None,
    bubble_mask: np.ndarray | None = None,
) -> tuple[TextMask, ...]:
    """Dispatch to the registered strategy implementation."""
    if recipe.strategy == MaskStrategyId.CTD_UNET:
        if bubble_mask is None:
            # fallback: bubble_mask was not provided at materialize time
            recipe = select_strategy(group, image, bubble_mask=None)
            return materialize(recipe, group, members, image)
        return _CTD_UNET.build(group, members, image, bubble_mask)
    if recipe.strategy == MaskStrategyId.OBB_PER_LINE:
        return _OBB_PER_LINE.build(group, members, image)
    if recipe.strategy == MaskStrategyId.PIXEL_SEG:
        return _PIXEL_SEG.build(group, members, image)
    return _RECT_DILATE.build(group, members, image)
