"""Domain types — Python mirror of crates/inpaint/src/domain/.

These are the **single source of truth** for Python callers and msgpack
encoders. Wire format: msgpack consumed by `crates/inpaint`.

Any change here must be reflected in the Rust mirror and the plan
version bumped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Plain enums (mirror crates/inpaint/src/domain/plan.rs) ────────────────

BlockClass = Literal["sfx", "dialogue", "narration"]
PageKind   = Literal["bw", "color", "webtoon"]
ShapeKind  = Literal["dialogue", "burst"]


# ── Wire format version ──────────────────────────────────────────────────
#
# Rust rejects any plan whose `version` field doesn't match.

PLAN_VERSION = 2


# ── Mask contract (mirror crates/inpaint/src/domain/group.rs) ─────────────

@dataclass(frozen=True, slots=True)
class EraseRaster:
    x:    int
    y:    int
    w:    int
    h:    int
    data: bytes   # zlib-compressed w*h u8 pixels, 0 or 255


# MaskKind is a tagged union encoded as a dict on the wire:
#   {"kind": "precise", "raster": {...}}
#   {"kind": "coarse",  "raster": {...}, "dilate_px": 4}
#   {"kind": "regen"}

@dataclass(frozen=True, slots=True)
class Precise:
    raster: EraseRaster

    def to_wire(self) -> dict:
        return {"kind": "precise", "raster": _raster_to_wire(self.raster)}


@dataclass(frozen=True, slots=True)
class Coarse:
    raster:     EraseRaster
    dilate_px:  int

    def to_wire(self) -> dict:
        return {
            "kind":      "coarse",
            "raster":    _raster_to_wire(self.raster),
            "dilate_px": int(self.dilate_px),
        }


@dataclass(frozen=True, slots=True)
class Regen:
    def to_wire(self) -> dict:
        return {"kind": "regen"}


MaskKind = Precise | Coarse | Regen


def _raster_to_wire(r: EraseRaster) -> dict:
    return {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "data": r.data}


# ── Group + plan ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Group:
    idx:        int
    bbox:       tuple[int, int, int, int]   # x1, y1, x2, y2
    class_:     BlockClass
    shape_kind: ShapeKind
    mask:       MaskKind


@dataclass(frozen=True, slots=True)
class InpaintPlan:
    page_index: int
    page_size:  tuple[int, int]    # W, H
    page_kind:  PageKind
    groups:     tuple[Group, ...]


# ── Coarse-mask dilate policy (vision-side, single source of truth) ──────
#
# Inpaint receives the resulting `dilate_px` baked into the Coarse
# variant; it has zero policy of its own. ~40 % of glyph short edge gives
# a 14 px dialogue line ~5 px breathing room and a 30 px SFX glyph ~10 px.

_DILATE_FRAC = 0.40
_DILATE_MIN  = 2
_DILATE_MAX  = 10


def dilate_for_glyph(glyph_px: int) -> int:
    """Outward dilate radius (px) for a Coarse raster, given glyph short edge."""
    return max(_DILATE_MIN, min(_DILATE_MAX, round(glyph_px * _DILATE_FRAC)))
