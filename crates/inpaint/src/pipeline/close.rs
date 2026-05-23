// SPDX-License-Identifier: GPL-3.0-or-later
//! Morphological close (dilate → erode) + enclosed-hole fill on a binary patch.

use crate::adapters::canny::fill_enclosed_holes;

/// Dilate then erode a 0/1 patch with a rectangular kernel of radius `r`,
/// then fill enclosed holes.  Returns 0/1 values (NOT 0/255).
pub fn close_and_fill(
    mut patch: Vec<u8>,
    pw:        usize,
    ph:        usize,
    r:         usize,
) -> anyhow::Result<Vec<u8>> {
    if r > 0 {
        patch = dilate_rect(patch, pw, ph, r);
        patch = erode_rect(patch, pw, ph, r);
    }
    fill_enclosed_holes(&mut patch, pw, ph);
    Ok(patch)
}

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
