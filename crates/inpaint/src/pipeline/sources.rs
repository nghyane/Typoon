// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-group mask executor + page mask accumulator.
//!
//! Pure dispatch on `MaskKind`. Inpaint owns no mask policy: vision
//! decides whether a raster is precise or coarse and (for coarse) how
//! much breathing room to add. We just run the kernel.

use anyhow::Result;

use crate::adapters::{rasterise, stroke_detect};
use crate::domain::{Group, InpaintPlan, MaskKind};
use crate::pipeline::close;
use super::DebugSink;

/// Build the per-group raster at page coords (`bbox.width × bbox.height`).
/// Returns 0/1 values.
pub fn build_group_patch(
    g:    &Group,
    rgb:  &[u8],
    w:    usize,
    h:    usize,
    page_kind: crate::domain::PageKind,
) -> Result<Vec<u8>> {
    let pw = g.bbox.width()  as usize;
    let ph = g.bbox.height() as usize;

    match &g.mask {
        MaskKind::Precise { raster } => {
            rasterise::rasters_to_patch(std::slice::from_ref(raster), &g.bbox)
        }
        MaskKind::Coarse { raster, dilate_px } => {
            let patch = rasterise::rasters_to_patch(std::slice::from_ref(raster), &g.bbox)?;
            Ok(close::dilate(patch, pw, ph, *dilate_px as usize))
        }
        MaskKind::Regen => {
            // Stroke-regen mask from page pixels inside bbox. Vision had
            // nothing usable; we run our own Canny + close-and-fill.
            let page_mask = stroke_detect::detect_strokes(rgb, w, h, g.bbox, page_kind)?;
            let x0 = g.bbox.x1 as usize;
            let y0 = g.bbox.y1 as usize;
            let raw: Vec<u8> = (0..ph).flat_map(|y| (0..pw).map(move |x| (x, y)))
                .map(|(x, y)| page_mask[(y0 + y) * w + (x0 + x)])
                .collect();
            // Light close to bridge stroke gaps. Radius scales with text size.
            let r = ((g.bbox.short_edge() as f32 * 0.06).round() as usize).max(2);
            close::close_and_fill(raw, pw, ph, r)
        }
    }
}

/// Build full-page 0/255 mask from all groups. Respects the "outsider
/// guard": a closed patch pixel is suppressed if it falls inside another
/// group's bbox AND the original raw page-mask at that position was 0
/// (prevents one group's dilate from bleeding into a neighbour's bbox).
pub fn build_page_mask(
    rgb:  &[u8],
    w:    usize,
    h:    usize,
    plan: &InpaintPlan,
    dbg:  &DebugSink,
) -> Result<Vec<u8>> {
    let mut result = vec![0u8; w * h];

    for g in &plan.groups {
        let patch = build_group_patch(g, rgb, w, h, plan.page_kind)?;

        let pw = g.bbox.width()  as usize;
        let ph = g.bbox.height() as usize;
        let x0 = g.bbox.x1 as usize;
        let y0 = g.bbox.y1 as usize;

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

        dbg.mask(
            &format!("group_{:03}_{}", g.idx, g.mask.tag()),
            &patch.iter().map(|&v| v * 255).collect::<Vec<_>>(),
            pw, ph,
        );
    }

    Ok(result)
}
