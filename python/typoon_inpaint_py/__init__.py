"""typoon_inpaint_py — pure-Python wrappers around the Rust PyO3 extension.

Import the compiled extension:
    from typoon_inpaint import InpaintRuntime, decode_plan, rasterise_plan_mask
"""
from __future__ import annotations

try:
    from typoon_inpaint.typoon_inpaint import (   # compiled extension
        InpaintRuntime,
        decode_plan,
        rasterise_plan_mask,
    )
except ImportError:
    raise ImportError(
        "typoon_inpaint native extension not found. "
        "Run `maturin develop --release` from the repo root."
    )

__all__ = ["InpaintRuntime", "decode_plan", "rasterise_plan_mask"]
