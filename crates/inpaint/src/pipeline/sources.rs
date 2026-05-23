// SPDX-License-Identifier: GPL-3.0-or-later
//! MaskSource strategy dispatch.
//!
//! Each `MaskOrigin` maps to exactly one source implementation.
//! The `match` is exhaustive — adding a new variant forces all sites to
//! be updated at compile time.

use anyhow::Result;

use crate::adapters::{canny, rasterise};
use crate::domain::{BBox, GroupMask, InpaintPlan, MaskOrigin, profile_for};
use crate::pipeline::close;
use super::DebugSink;

// ── Per-group raw mask ───────────────────────────────────────────────────

/// Build raw 0/1 raster for one group via the correct strategy, then
/// apply morphological close + hole-fill.
// ── Page mask accumulation ────────────────────────────────────────────────

/// Build full-page 0/255 mask from all groups. Respects the "outsider
/// guard": a closed patch pixel is suppressed if it falls inside another
/// group's bbox AND the original raw mask at that position was 0.
pub fn build_page_mask(
    rgb:  &[u8],
    w:    usize,
    h:    usize,
    plan: &InpaintPlan,
    dbg:  &DebugSink,
) -> Result<Vec<u8>> {
    let mut result = vec![0u8; w * h];

    for g in &plan.groups {
        let patch = build_group_patch_with_kind(g, rgb, w, h, plan.page_kind)?;

        let pw = g.bbox.width()  as usize;
        let ph = g.bbox.height() as usize;
        let x0 = g.bbox.x1 as usize;
        let y0 = g.bbox.y1 as usize;

        // Outsider guard: if another group's centre is inside this patch's
        // expanded region, don't write closed=1 pixels that weren't already 1
        // in the raw result.
        let bridges = plan.groups.iter().any(|o| {
            if o.idx == g.idx { return false; }
            let (cx, cy) = o.bbox.centre();
            cx >= x0 as f32 && cx < (x0 + pw) as f32 &&
            cy >= y0 as f32 && cy < (y0 + ph) as f32
        });

        let raw_before: Vec<u8> = (0..ph).flat_map(|y| (0..pw).map(move |x| (x, y)))
            .map(|(x, y)| result[(y0 + y) * w + (x0 + x)])
            .collect();

        for y in 0..ph {
            for x in 0..pw {
                if patch[y * pw + x] == 0 { continue; }
                let pi = (y0 + y) * w + (x0 + x);
                if raw_before[y * pw + x] == 0 && bridges { continue; }
                result[pi] = 255;
            }
        }

        dbg.mask(&format!("group_{:03}_patch", g.idx),
                 &patch.iter().map(|&v| v * 255).collect::<Vec<_>>(),
                 pw, ph);
    }

    Ok(result)
}

fn build_group_patch_with_kind(
    g:     &GroupMask,
    rgb:   &[u8],
    img_w: usize,
    img_h: usize,
    page_kind: crate::domain::PageKind,
) -> Result<Vec<u8>> {
    let raw: Vec<u8> = match g.origin {
        MaskOrigin::LensObb | MaskOrigin::LensAabb =>
            rasterise::polygons_to_patch(&g.polygons, &g.bbox),
        MaskOrigin::CtdUnet =>
            rasterise::rasters_to_patch(&g.rasters, &g.bbox)?,
        MaskOrigin::PolygonFallback => {
            let page_mask = crate::adapters::stroke_detect::detect_strokes(
                rgb, img_w, img_h, g.bbox, page_kind,
            )?;
            let pw = g.bbox.width()  as usize;
            let ph = g.bbox.height() as usize;
            let x0 = g.bbox.x1 as usize;
            let y0 = g.bbox.y1 as usize;
            (0..ph).flat_map(|y| (0..pw).map(move |x| (x, y)))
                .map(|(x, y)| page_mask[(y0 + y) * img_w + (x0 + x)])
                .collect()
        }
    };

    let profile = profile_for(g.class);
    let r = profile.close_radius(g.bbox.short_edge());
    close::close_and_fill(raw, g.bbox.width() as usize, g.bbox.height() as usize, r as usize)
}
