// SPDX-License-Identifier: GPL-3.0-or-later
//! Polygon → binary raster and CTD raster paste.

use anyhow::{Result, anyhow};

use crate::domain::{BBox, EraseRaster};

/// Scanline-fill a convex or non-convex polygon into a patch buffer.
/// Polygon vertices are `[[x, y], ...]` in page coords.
/// Output: flat `0/1` buffer of size `(bbox.width() × bbox.height())`.
pub fn polygons_to_patch(
    polygons: &[Vec<[f32; 2]>],
    bbox:     &BBox,
) -> Vec<u8> {
    let pw = bbox.width()  as usize;
    let ph = bbox.height() as usize;
    let ox = bbox.x1 as f32;
    let oy = bbox.y1 as f32;
    let mut patch = vec![0u8; pw * ph];

    for poly in polygons {
        if poly.len() < 3 { continue; }
        scanline_fill(poly, ox, oy, pw, ph, &mut patch);
    }
    patch
}

/// Paste CTD rasters into a patch aligned to `bbox`.
/// Rasters may overlap — OR them.
pub fn rasters_to_patch(
    rasters: &[EraseRaster],
    bbox:    &BBox,
) -> Result<Vec<u8>> {
    let pw = bbox.width()  as usize;
    let ph = bbox.height() as usize;
    let mut patch = vec![0u8; pw * ph];

    for r in rasters {
        let expected = r.w as usize * r.h as usize;
        if r.data.len() != expected {
            return Err(anyhow!(
                "raster data len {} != {}×{}={}", r.data.len(), r.w, r.h, expected
            ));
        }
        let dx = r.x - bbox.x1;
        let dy = r.y - bbox.y1;
        for ry in 0..r.h as i32 {
            for rx in 0..r.w as i32 {
                let px = dx + rx;
                let py = dy + ry;
                if px < 0 || py < 0 || px >= pw as i32 || py >= ph as i32 { continue; }
                let src = r.data[(ry * r.w as i32 + rx) as usize];
                if src >= 127 {
                    patch[py as usize * pw + px as usize] = 1;
                }
            }
        }
    }
    Ok(patch)
}

// ── Scanline polygon fill ─────────────────────────────────────────────────
// Even-odd rule. Works for convex and simple concave polygons.

fn scanline_fill(
    poly: &[[f32; 2]],
    ox: f32, oy: f32,
    pw: usize, ph: usize,
    out: &mut [u8],
) {
    let n = poly.len();

    // Local coords
    let local: Vec<(f32, f32)> = poly.iter()
        .map(|&[x, y]| (x - ox, y - oy))
        .collect();

    let y_min = local.iter().map(|&(_, y)| y).fold(f32::INFINITY, f32::min).max(0.0) as usize;
    let y_max = local.iter().map(|&(_, y)| y).fold(f32::NEG_INFINITY, f32::max).min(ph as f32 - 1.0) as usize;

    for y in y_min..=y_max {
        let yf = y as f32 + 0.5;
        let mut xs = Vec::new();
        for i in 0..n {
            let (x0, y0) = local[i];
            let (x1, y1) = local[(i + 1) % n];
            if (y0 <= yf && yf < y1) || (y1 <= yf && yf < y0) {
                let t = (yf - y0) / (y1 - y0);
                xs.push(x0 + t * (x1 - x0));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i + 1 < xs.len() {
            let x0 = xs[i].max(0.0) as usize;
            let x1 = xs[i+1].min(pw as f32 - 1.0) as usize;
            for x in x0..=x1 { out[y * pw + x] = 1; }
            i += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_polygon_fills_interior() {
        let bbox  = BBox::new(10, 10, 30, 30).unwrap();
        let poly  = vec![[10.0f32, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]];
        let patch = polygons_to_patch(&[poly], &bbox);
        // All pixels inside 20×20 patch should be 1
        assert!(patch.iter().all(|&v| v == 1), "interior not fully filled");
    }

    #[test]
    fn triangle_partial_fill() {
        let bbox = BBox::new(0, 0, 10, 10).unwrap();
        let poly = vec![[0.0f32, 0.0], [10.0, 0.0], [5.0, 10.0]];
        let patch = polygons_to_patch(&[poly], &bbox);
        // Top-left corner (0,0) in patch but bottom-left corner (0,9) should be 0
        assert_eq!(patch[0], 1);         // (0,0) inside triangle top edge
        assert_eq!(patch[9 * 10 + 0], 0); // (0,9) outside
    }
}
