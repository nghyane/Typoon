"""typoon_inpaint — Rust-backed inpaint pipeline + Python utilities.

The compiled PyO3 extension is at typoon_inpaint.typoon_inpaint.
Pure-Python submodules: domain, scan, pipeline, storage, artifact_sink, cli.
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
