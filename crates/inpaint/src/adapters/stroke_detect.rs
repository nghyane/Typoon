// SPDX-License-Identifier: GPL-3.0-or-later
//! Stroke detection: find ink pixels within an OBB region.
//!
//! Used for `polygon_fallback` groups where no erase_mask was available.
//! Also used as a refine pass when the inpaint stage wants tighter masks.
//!
//! Algorithm: adaptive luminance threshold based on background lum, then
//! morphological close to fill stroke gaps, then enclosed-hole flood fill.
//!
//! Handles:
//!   - Dark ink on white/light bg (manga standard)
//!   - White/light ink on dark bg (title pages, inverted panels)
//!   - Colored ink on any bg (color manga)

use anyhow::Result;
use crate::domain::{BBox, PageKind};

const PAD_FRAC: f32 = 0.15;
const PAD_MIN:  i32 = 8;

// ── Tuning ───────────────────────────────────────────────────────────────

const DARK_INK_LUM:    i16 = 80;   // lum < this → dark ink
const LIGHT_INK_LUM:   i16 = 180;  // lum > this → light ink (when bg is dark)
const DARK_BG_THRESH:  f32 = 128.0;// median lum below this → bg is dark
const COLOR_SAT_MIN:   i16 = 40;   // saturation threshold for colored ink
const COLOR_CONTRAST:  i16 = 30;   // R-B or other channel diff
// Morphological close to fill stroke gaps (pixels)
const CLOSE_RADIUS:    usize = 2;

// ── Public entry ─────────────────────────────────────────────────────────

/// Detect ink strokes in the given bbox region of `img`.
///
/// `img` — flat RGB, 3 bytes/pixel, W×H.
/// Returns a full-page 0/1 mask (same W×H); only bbox region is set.
pub fn detect_strokes(
    img:       &[u8],
    img_w:     usize,
    img_h:     usize,
    bbox:      BBox,
    page_kind: PageKind,
) -> Result<Vec<u8>> {
    let short = bbox.short_edge() as f32;
    let pad   = ((short * PAD_FRAC).round() as i32).max(PAD_MIN);
    let crop  = bbox.expand(pad, img_w as i32, img_h as i32);
    let cw    = crop.width()  as usize;
    let ch    = crop.height() as usize;
    let cx0   = crop.x1 as usize;
    let cy0   = crop.y1 as usize;

    // 1. Extract luminance + saturation for crop
    let mut lum = vec![0i16; cw * ch];
    let mut sat = vec![0i16; cw * ch];
    let mut rb_diff = vec![0i16; cw * ch];

    for y in 0..ch {
        for x in 0..cw {
            let si = ((cy0 + y) * img_w + (cx0 + x)) * 3;
            let r = img[si]   as i16;
            let g = img[si+1] as i16;
            let b = img[si+2] as i16;
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            lum[y*cw+x]     = (0.299*r as f32 + 0.587*g as f32 + 0.114*b as f32) as i16;
            sat[y*cw+x]     = mx - mn;
            rb_diff[y*cw+x] = r - b;
        }
    }

    // 2. Estimate background luminance (median via percentile approximation)
    let mut sorted_lum = lum.clone();
    sorted_lum.sort_unstable();
    let bg_lum = sorted_lum[sorted_lum.len() / 2] as f32;

    // 3. Classify ink pixels
    let mut mask = vec![0u8; cw * ch];
    for i in 0..cw * ch {
        let l = lum[i];
        let s = sat[i];
        let d = rb_diff[i];

        let is_ink = match page_kind {
            PageKind::Bw | PageKind::Webtoon => {
                if bg_lum > DARK_BG_THRESH {
                    // Light bg → dark ink
                    l < DARK_INK_LUM
                } else {
                    // Dark bg → light ink
                    l > LIGHT_INK_LUM
                }
            }
            PageKind::Color => {
                // Dark ink
                (l < DARK_INK_LUM)
                // White ink on dark bg
                || (l > LIGHT_INK_LUM && bg_lum < DARK_BG_THRESH)
                // Colored ink (saturated, high contrast)
                || (s > COLOR_SAT_MIN && (d > COLOR_CONTRAST || -d > COLOR_CONTRAST))
            }
        };
        if is_ink { mask[i] = 1; }
    }

    // 4. Morphological close: fill small gaps between stroke pixels
    if CLOSE_RADIUS > 0 {
        mask = dilate_rect(mask, cw, ch, CLOSE_RADIUS);
        mask = erode_rect(mask, cw, ch, CLOSE_RADIUS);
    }

    // 5. Flood-fill enclosed holes from border
    fill_enclosed_holes(&mut mask, cw, ch);

    // 6. Embed into full-page mask
    let mut out = vec![0u8; img_w * img_h];
    for y in 0..ch {
        for x in 0..cw {
            out[(cy0 + y) * img_w + (cx0 + x)] = mask[y * cw + x];
        }
    }
    Ok(out)
}

// ── Morphological helpers ─────────────────────────────────────────────────

fn dilate_rect(src: Vec<u8>, w: usize, h: usize, r: usize) -> Vec<u8> {
    let mut h1 = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            if (lo..=hi).any(|xx| src[row + xx] != 0) { h1[row + x] = 1; }
        }
    }
    let mut out = vec![0u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            if (lo..=hi).any(|yy| h1[yy * w + x] != 0) { out[y * w + x] = 1; }
        }
    }
    out
}

fn erode_rect(src: Vec<u8>, w: usize, h: usize, r: usize) -> Vec<u8> {
    let mut h1 = vec![1u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            if (lo..=hi).any(|xx| src[row + xx] == 0) { h1[row + x] = 0; }
        }
    }
    let mut out = vec![1u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            if (lo..=hi).any(|yy| h1[yy * w + x] == 0) { out[y * w + x] = 0; }
        }
    }
    out
}

// ── Flood-fill enclosed holes (shared with Canny path) ───────────────────

pub fn fill_enclosed_holes(mask: &mut Vec<u8>, w: usize, h: usize) {
    if w == 0 || h == 0 { return; }
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();

    macro_rules! push {
        ($idx:expr) => {
            let i = $idx;
            if mask[i] == 0 && !outside[i] { outside[i] = true; stack.push(i); }
        };
    }

    for x in 0..w { push!(x); push!((h-1)*w+x); }
    for y in 0..h { push!(y*w); push!(y*w+(w-1)); }

    while let Some(i) = stack.pop() {
        let x = i % w; let y = i / w;
        if x > 0     { push!(i-1); }
        if x+1 < w   { push!(i+1); }
        if y > 0     { push!(i-w); }
        if y+1 < h   { push!(i+w); }
    }

    for i in 0..w*h {
        if mask[i] == 0 && !outside[i] { mask[i] = 1; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_ink_on_white_bg_detected() {
        // 10×10 image: white bg with dark center 4×4
        let mut img = vec![255u8; 10 * 10 * 3];
        for y in 3..7 { for x in 3..7 {
            let i = (y * 10 + x) * 3;
            img[i] = 30; img[i+1] = 30; img[i+2] = 30;
        }}
        let bbox = BBox::new(2, 2, 8, 8).unwrap();
        let mask = detect_strokes(&img, 10, 10, bbox, PageKind::Bw).unwrap();
        // Dark center pixels should be ink
        assert!(mask[4 * 10 + 4] > 0, "center ink pixel not detected");
        // White corner should not be ink
        assert_eq!(mask[0], 0, "white corner falsely detected as ink");
    }

    #[test]
    fn white_ink_on_dark_bg_detected() {
        // 10×10 image: dark bg with white center
        let mut img = vec![30u8; 10 * 10 * 3];
        for y in 3..7 { for x in 3..7 {
            let i = (y * 10 + x) * 3;
            img[i] = 240; img[i+1] = 240; img[i+2] = 240;
        }}
        let bbox = BBox::new(2, 2, 8, 8).unwrap();
        let mask = detect_strokes(&img, 10, 10, bbox, PageKind::Bw).unwrap();
        assert!(mask[4 * 10 + 4] > 0, "white ink not detected on dark bg");
    }

    #[test]
    fn hole_fill_closed_ring() {
        let mut m = vec![0u8; 25]; // 5×5
        for x in 1..4 { m[5+x]=1; m[15+x]=1; }
        for y in 1..4 { m[y*5+1]=1; m[y*5+3]=1; }
        fill_enclosed_holes(&mut m, 5, 5);
        assert_eq!(m[12], 1, "enclosed hole not filled");
        assert_eq!(m[0],  0, "border pixel incorrectly filled");
    }
}
