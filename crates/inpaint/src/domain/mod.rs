// SPDX-License-Identifier: GPL-3.0-or-later
//! Domain types: the single source of truth shared by every pipeline
//! stage.  No I/O, no rendering, no model calls here — only data
//! contracts and their invariants.
//!
//! Python mirror: `python/typoon_inpaint_py/domain.py` (frozen
//! dataclasses, same field names).  Wire format: msgpack produced by
//! the scan container, consumed here via `rmp-serde`.

mod bbox;
mod plan;
mod profile;

pub use bbox::BBox;
pub use plan::{EraseRaster, GroupMask, InpaintPlan};
pub use profile::{BlockClass, MaskOrigin, PadProfile, PageKind, ShapeKind, PROFILES, profile_for};
