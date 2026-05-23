"""Domain types — Python mirror of crates/inpaint/src/domain/.

These are the **single source of truth** for Python callers.
Wire format: msgpack produced by the scan container.

Any change here must be reflected in:
  crates/inpaint/src/domain/profile.rs  (Rust)
  tests/fixtures/profiles-golden.json   (golden test)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Enums ─────────────────────────────────────────────────────────────────

MaskOrigin = Literal["lens_obb", "lens_aabb", "ctd_unet", "polygon_fallback"]
BlockClass = Literal["sfx", "dialogue", "narration"]
PageKind   = Literal["bw", "color", "webtoon"]
ShapeKind  = Literal["dialogue", "burst"]


# ── Pad profile — one table, all layers ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class PadProfile:
    container_pad_frac: float
    container_pad_min:  int
    mask_pad_frac:      float
    mask_pad_min:       int
    close_radius_frac:  float
    close_radius_min:   int
    context_frac:       float

    def close_radius(self, short_edge: int) -> int:
        return max(self.close_radius_min,
                   round(short_edge * self.close_radius_frac))


PROFILES: dict[BlockClass, PadProfile] = {
    "sfx":       PadProfile(0.08, 4, 0.08, 2, 0.15, 2, 0.60),
    "dialogue":  PadProfile(0.20, 4, 0.20, 2, 0.10, 2, 0.50),
    "narration": PadProfile(0.20, 4, 0.20, 2, 0.12, 2, 0.50),
}


# ── Wire types ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EraseRaster:
    x:    int
    y:    int
    w:    int
    h:    int
    data: bytes  # w*h bytes, 0 or 255


@dataclass(frozen=True, slots=True)
class GroupMask:
    idx:        int
    bbox:       tuple[int, int, int, int]   # x1, y1, x2, y2
    origin:     MaskOrigin
    class_:     BlockClass
    shape_kind: ShapeKind
    polygons:   tuple[list[tuple[float, float]], ...] = field(default_factory=tuple)
    rasters:    tuple[EraseRaster, ...]               = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.origin in ("lens_obb", "lens_aabb") and not self.polygons and not self.rasters:
            raise ValueError(f"group {self.idx}: {self.origin} requires polygons or rasters")
        if self.origin == "ctd_unet" and not self.rasters:
            raise ValueError(f"group {self.idx}: ctd_unet requires rasters")
        if self.origin == "polygon_fallback" and (self.polygons or self.rasters):
            raise ValueError(
                f"group {self.idx}: polygon_fallback must not ship pixel data")


@dataclass(frozen=True, slots=True)
class InpaintPlan:
    page_index: int
    page_size:  tuple[int, int]   # W, H
    page_kind:  PageKind
    groups:     tuple[GroupMask, ...]
