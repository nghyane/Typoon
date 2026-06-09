// SPDX-License-Identifier: GPL-3.0-or-later
//! Mask contract between vision (producer) and inpaint (executor).
//!
//! Vision tags each group with a `MaskKind` describing what inpaint must
//! do with the raster. Inpaint is a pure executor: no policy lookup, no
//! per-class profile table. Producer-side knowledge (glyph size, pre-
//! padding state) is baked into the variant payload at emit time.
//!
//! Python mirror: `python/typoon_inpaint/domain.py`.

use serde::Deserialize;

use super::{BBox, BlockClass, ShapeKind};

/// Tight raster shipped inside a `MaskKind` variant.
/// `data` is zlib-compressed raw 1-channel u8 (`w*h` bytes uncompressed,
/// 0 = background, 255 = ink). Always decompress before use.
#[derive(Debug, Deserialize)]
pub struct EraseRaster {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl EraseRaster {
    pub fn decompress(&self) -> anyhow::Result<Vec<u8>> {
        use std::io::Read;
        let mut dec = flate2::read::ZlibDecoder::new(self.data.as_slice());
        let mut out = Vec::with_capacity((self.w * self.h) as usize);
        dec.read_to_end(&mut out)
            .map_err(|e| anyhow::anyhow!("raster decompress: {e}"))?;
        if out.len() != (self.w * self.h) as usize {
            anyhow::bail!(
                "raster decompress: expected {}×{}={} bytes, got {}",
                self.w, self.h, self.w * self.h, out.len()
            );
        }
        Ok(out)
    }
}

/// What inpaint must do with this group's mask. Exhaustive — adding a
/// variant forces every dispatch site to compile.
///
/// Wire format (msgpack, internally tagged on field `kind`):
/// ```text
/// { "kind": "precise", "raster": {...} }
/// { "kind": "coarse",  "raster": {...}, "dilate_px": 4 }
/// { "kind": "regen" }
/// ```
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskKind {
    /// Raster is pixel-aligned with glyphs (CTD UNet crop, Otsu+morph
    /// pixel seg). Use as-is; no morphological ops.
    Precise { raster: EraseRaster },

    /// Raster is a coarse line/word stripe. Inpaint dilates outward by
    /// `dilate_px` to add breathing room. Vision picks the value based
    /// on glyph size — inpaint just runs the kernel.
    Coarse  { raster: EraseRaster, dilate_px: u8 },

    /// No raster shipped. Inpaint regenerates the mask by stroke
    /// detection inside `bbox` (Canny + close-and-fill).
    Regen,
}

impl MaskKind {
    /// Short tag for debug / probe counters.
    pub fn tag(&self) -> &'static str {
        match self {
            MaskKind::Precise { .. } => "precise",
            MaskKind::Coarse  { .. } => "coarse",
            MaskKind::Regen          => "regen",
        }
    }
}

/// One group as it appears in the InpaintPlan.
#[derive(Debug, Deserialize)]
pub struct Group {
    pub idx:        i32,
    pub bbox:       BBox,
    pub class:      BlockClass,
    pub shape_kind: ShapeKind,
    pub mask:       MaskKind,
}
