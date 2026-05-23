// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-class pad profiles — single source of truth for every layer.
//!
//! Vision (Python) reads the same values from `python/typoon_inpaint_py/domain.py`.
//! Any tuning change must be reflected in both files + golden fixture test.

use serde::Deserialize;

// ── Tagged enums ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskOrigin {
    /// Vision produced tight per-line OBB stripe(s).
    LensObb,
    /// Vision produced AABB (column layout / single-word / no-line).
    LensAabb,
    /// CTD UNet pixel mask embedded as raster.
    CtdUnet,
    /// No erase masks available — bbox rectangle was used as fallback.
    /// Inpaint must regenerate mask via Canny stroke detection.
    PolygonFallback,
}

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
    /// Monochrome manga — luminance Canny sufficient.
    Bw,
    /// Full colour page — LAB ΔE supplement for Canny.
    Color,
    /// Webtoon vertical strip — region merge distance scaled.
    Webtoon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeKind {
    Dialogue,
    Burst,
}

// ── Pad profile — one struct, three stage consumers ──────────────────────

/// Parameters that govern mask and tile sizing for one block class.
///
/// Consumers:
///   - Vision (Python): `container_pad_frac` + `mask_pad_frac`
///   - Inpaint close: `close_radius_frac` + `close_radius_min`
///   - Tile context grow: `context_frac`
///
/// Changing any value here must be reflected in:
///   `python/typoon_inpaint_py/domain.py` PROFILES
///   `tests/fixtures/profiles.json`  (golden roundtrip)
#[derive(Debug, Clone, Copy)]
pub struct PadProfile {
    /// Fraction of group short-edge used for polygon container expand.
    pub container_pad_frac: f64,
    pub container_pad_min:  i32,
    /// Fraction of group short-edge used for erase mask expand.
    pub mask_pad_frac:      f64,
    pub mask_pad_min:       i32,
    /// Fraction of group short-edge used for morphological close radius.
    pub close_radius_frac:  f64,
    pub close_radius_min:   i32,
    /// Fraction of short-edge added as context around the AOT tile.
    pub context_frac:       f32,
}

impl PadProfile {
    pub fn close_radius(&self, short_edge: i32) -> i32 {
        ((short_edge as f64 * self.close_radius_frac).round() as i32)
            .max(self.close_radius_min)
    }
}

/// Compile-time table indexed by `BlockClass`.
/// `const fn` means the compiler inlines at every call site.
pub const fn profile_for(c: BlockClass) -> PadProfile {
    match c {
        BlockClass::Sfx       => SFX_PROFILE,
        BlockClass::Dialogue  => DIALOGUE_PROFILE,
        BlockClass::Narration => NARRATION_PROFILE,
    }
}

const SFX_PROFILE: PadProfile = PadProfile {
    container_pad_frac: 0.08, container_pad_min: 4,
    mask_pad_frac:      0.08, mask_pad_min:      2,
    close_radius_frac:  0.15, close_radius_min:  2,
    context_frac:       0.60,
};

const DIALOGUE_PROFILE: PadProfile = PadProfile {
    container_pad_frac: 0.20, container_pad_min: 4,
    mask_pad_frac:      0.20, mask_pad_min:      2,
    close_radius_frac:  0.10, close_radius_min:  2,
    context_frac:       0.50,
};

const NARRATION_PROFILE: PadProfile = PadProfile {
    container_pad_frac: 0.20, container_pad_min: 4,
    mask_pad_frac:      0.20, mask_pad_min:      2,
    close_radius_frac:  0.12, close_radius_min:  2,
    context_frac:       0.50,
};

// Re-export as array so Python test can check JSON fixture.
pub const PROFILES: [(&str, PadProfile); 3] = [
    ("sfx",       SFX_PROFILE),
    ("dialogue",  DIALOGUE_PROFILE),
    ("narration", NARRATION_PROFILE),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_radius_sfx() {
        let p = profile_for(BlockClass::Sfx);
        // 200px short edge → 200 * 0.15 = 30
        assert_eq!(p.close_radius(200), 30);
        // very small → clamped to min=2
        assert_eq!(p.close_radius(0), 2);
    }

    #[test]
    fn close_radius_dialogue() {
        let p = profile_for(BlockClass::Dialogue);
        // 100px short → 100 * 0.10 = 10
        assert_eq!(p.close_radius(100), 10);
    }

    #[test]
    fn close_radius_narration() {
        let p = profile_for(BlockClass::Narration);
        // 50px → 50 * 0.12 = 6
        assert_eq!(p.close_radius(50), 6);
    }
}
