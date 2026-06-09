// SPDX-License-Identifier: GPL-3.0-or-later
//! Domain types: the single source of truth shared by every pipeline
//! stage.  No I/O, no rendering, no model calls here — only data
//! contracts and their invariants.
//!
//! Python mirror: `python/typoon_inpaint/domain.py` (frozen
//! dataclasses, same field names).  Wire format: msgpack produced by
//! the scan container, consumed here via `rmp-serde`.

mod bbox;
mod group;
mod plan;

pub use bbox::BBox;
pub use group::{EraseRaster, Group, MaskKind};
pub use plan::{BlockClass, InpaintPlan, PageKind, ShapeKind, PLAN_VERSION};
