// SPDX-License-Identifier: GPL-3.0-or-later
//! Wire types for the InpaintPlan msgpack produced by the scan container.
//!
//! Schema version: v2  (filename `scan/{job}/{i:04d}.msgpack`)
//! Python source:  `python/typoon_inpaint/domain.py`
//!
//! v1 → v2 break:
//!   - `origin: MaskOrigin` + `polygons` + `rasters` flat fields
//!     replaced by `mask: MaskKind` tagged union.
//!   - PadProfile table removed from Rust entirely; vision owns policy.

use serde::Deserialize;

use super::Group;

/// Plain enums shared with vision. No policy attached — those live
/// in vision Python (`PROFILES` in `domain.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockClass {
    Sfx,
    Dialogue,
    Narration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageKind {
    Bw,
    Color,
    Webtoon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeKind {
    Dialogue,
    Burst,
}

/// Wire-version of the plan we accept. Bump on every breaking change to
/// the contract; reject everything else loudly.
pub const PLAN_VERSION: u32 = 2;

/// Top-level plan decoded from `scan/{job}/{page:04d}.msgpack`.
#[derive(Debug, Deserialize)]
pub struct InpaintPlan {
    pub version:    u32,
    pub page_index: u32,
    pub page_size:  [u32; 2],
    pub page_kind:  PageKind,
    pub groups:     Vec<Group>,
}

impl InpaintPlan {
    /// Decode from msgpack bytes.
    ///
    /// Accepts two wire forms:
    ///   1. Bare InpaintPlan msgpack.
    ///   2. Scan msgpack with embedded `inpaint_plan` bytes.
    pub fn from_msgpack(bytes: &[u8]) -> anyhow::Result<Self> {
        let plan: Self = match rmp_serde::from_slice::<Self>(bytes) {
            Ok(p) => p,
            Err(_) => {
                #[derive(serde::Deserialize)]
                struct Wrapper {
                    #[serde(with = "serde_bytes")]
                    inpaint_plan: Vec<u8>,
                }
                let w: Wrapper = rmp_serde::from_slice(bytes).map_err(|e| {
                    anyhow::anyhow!("InpaintPlan decode (direct + scan-embedded): {e}")
                })?;
                rmp_serde::from_slice(&w.inpaint_plan)
                    .map_err(|e| anyhow::anyhow!("InpaintPlan embedded decode: {e}"))?
            }
        };
        if plan.version != PLAN_VERSION {
            anyhow::bail!(
                "InpaintPlan version mismatch: got {}, expected {} \
                 (regenerate scan msgpack)",
                plan.version, PLAN_VERSION
            );
        }
        Ok(plan)
    }

    pub fn page_w(&self) -> u32 { self.page_size[0] }
    pub fn page_h(&self) -> u32 { self.page_size[1] }
}
