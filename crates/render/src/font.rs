//! Embedded font + lazy skrifa/harfrust handles.
//!
//! Both shaping (HarfBuzz) and outlining (skrifa) operate from the same
//! byte slice. Keeping a single source of truth removes the scale mismatch
//! that bit the old `harfrust + ab_glyph` pipeline (HarfBuzz advances were
//! em-based while ab_glyph rasterised at `px / height_unscaled`, producing
//! the bloated word spacing on tall fonts like SamaritanTall-TB).
//!
//! Now everything is em-based:
//!   * HarfBuzz `point_size(N)` ⇒ advance scale `N / upem`
//!   * skrifa  `Size::new(N)`   ⇒ outline scale `N / upem`
//!
//! No workaround scaling needed at any consumer site.

use std::sync::OnceLock;

use harfrust::FontRef as HbFontRef;
use skrifa::FontRef as SkFontRef;

pub const FONT_BYTES: &[u8] = include_bytes!("../assets/SamaritanTall-TB.ttf");

static SKRIFA_FONT: OnceLock<SkFontRef<'static>> = OnceLock::new();

/// Skrifa font handle for outline drawing.
pub fn skrifa_font() -> &'static SkFontRef<'static> {
    SKRIFA_FONT.get_or_init(|| {
        SkFontRef::from_index(FONT_BYTES, 0).expect("skrifa: parse SamaritanTall-TB")
    })
}

/// HarfBuzz font handle. Cheap to construct, no benefit from caching since
/// `ShaperData` already gets built per call site and the underlying tables
/// are zero-copy reads.
pub fn harfrust_font() -> HbFontRef<'static> {
    HbFontRef::from_index(FONT_BYTES, 0).expect("harfrust: parse SamaritanTall-TB")
}

/// Units-per-em of the embedded font. Read once from skrifa head table.
pub fn units_per_em() -> u16 {
    use skrifa::raw::TableProvider;
    skrifa_font()
        .head()
        .map(|h| h.units_per_em())
        .unwrap_or(1000)
}
