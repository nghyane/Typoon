// SPDX-License-Identifier: GPL-3.0-or-later
//! Wire types for the InpaintPlan msgpack produced by the scan container.
//!
//! Schema version: v1  (filename `scan/{job}/{i:04d}.msgpack`)
//! Python source:  `python/typoon_inpaint_py/domain.py`

use serde::Deserialize;

use super::{BBox, BlockClass, MaskOrigin, PageKind, ShapeKind};

/// Tight raster emitted only when `mask_origin == ctd_unet`.
/// Embedded as raw bytes (1 byte / pixel, 0 or 255) in page coords.
#[derive(Debug, Deserialize)]
pub struct EraseRaster {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    /// Raw 1-channel u8 pixels, `w * h` bytes, 0 = background, 255 = ink.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// Per-group mask descriptor shipped in the InpaintPlan.
///
/// Invariants (checked by `GroupMask::validate`):
///   - `origin == lens_obb | lens_aabb` → `polygons` non-empty
///   - `origin == ctd_unet`             → `rasters` non-empty
///   - `origin == polygon_fallback`     → `polygons` and `rasters` empty
#[derive(Debug, Deserialize)]
pub struct GroupMask {
    pub idx:        i32,
    pub bbox:       BBox,
    pub origin:     MaskOrigin,
    pub class:      BlockClass,
    pub shape_kind: ShapeKind,
    /// Erase polygon stripes in page coords. Each inner vec is one polygon
    /// `[[x, y], ...]` with ≥ 3 vertices.
    #[serde(default)]
    pub polygons:   Vec<Vec<[f32; 2]>>,
    /// CTD UNet rasters — populated only when `origin == ctd_unet`.
    #[serde(default)]
    pub rasters:    Vec<EraseRaster>,
}

impl GroupMask {
    /// Validate invariants after deserialization.
    pub fn validate(&self) -> anyhow::Result<()> {
        use MaskOrigin::*;
        match self.origin {
            LensObb | LensAabb => {
                if self.polygons.is_empty() {
                    anyhow::bail!(
                        "group {}: origin={:?} requires non-empty polygons",
                        self.idx, self.origin
                    );
                }
            }
            CtdUnet => {
                if self.rasters.is_empty() {
                    anyhow::bail!(
                        "group {}: ctd_unet requires non-empty rasters",
                        self.idx
                    );
                }
            }
            PolygonFallback => {
                if !self.polygons.is_empty() || !self.rasters.is_empty() {
                    anyhow::bail!(
                        "group {}: polygon_fallback must not ship pixel data \
                         (inpaint will regen via Canny)",
                        self.idx
                    );
                }
            }
        }
        Ok(())
    }
}

/// Top-level plan decoded from `scan/{job}/{page:04d}.msgpack`.
#[derive(Debug, Deserialize)]
pub struct InpaintPlan {
    pub page_index: u32,
    pub page_size:  [u32; 2],    // [W, H]
    pub page_kind:  PageKind,
    pub groups:     Vec<GroupMask>,
}

impl InpaintPlan {
    /// Decode from msgpack bytes and validate all group invariants.
    ///
    /// Handles two wire formats:
    ///   1. Direct InpaintPlan msgpack (has `page_index`, `page_size`, `page_kind`, `groups`)
    ///   2. Scan msgpack (has `inpaint_plan` field containing embedded InpaintPlan bytes)
    pub fn from_msgpack(bytes: &[u8]) -> anyhow::Result<Self> {
        // Try to detect scan msgpack by checking for `inpaint_plan` key.
        // rmp_serde will fail on the outer map if it's not an InpaintPlan,
        // so we use a two-pass approach: try direct first, then scan-embedded.
        if let Ok(plan) = rmp_serde::from_slice::<Self>(bytes) {
            for g in &plan.groups { g.validate()?; }
            return Ok(plan);
        }
        // Try scan msgpack: deserialize as generic map, extract inpaint_plan bytes.
        #[derive(serde::Deserialize)]
        struct ScanWrapper {
            #[serde(with = "serde_bytes")]
            inpaint_plan: Vec<u8>,
        }
        let wrapper: ScanWrapper = rmp_serde::from_slice(bytes)
            .map_err(|e| anyhow::anyhow!("InpaintPlan decode (direct + scan-embedded): {e}"))?;
        let plan: Self = rmp_serde::from_slice(&wrapper.inpaint_plan)
            .map_err(|e| anyhow::anyhow!("InpaintPlan embedded decode: {e}"))?;
        for g in &plan.groups { g.validate()?; }
        Ok(plan)
    }

    pub fn page_w(&self) -> u32 { self.page_size[0] }
    pub fn page_h(&self) -> u32 { self.page_size[1] }
}
