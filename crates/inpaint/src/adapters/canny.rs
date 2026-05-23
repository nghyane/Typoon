// SPDX-License-Identifier: GPL-3.0-or-later
//! Canny-based stroke detection for `polygon_fallback` groups.
//!
//! Algorithm (per doc/wiki/inpaint-mask-generation.md):
//!   1. Crop image around bbox + padding (15% short edge, min 8 px).
//!   2. Convert RGB → luminance.
//!   3. Sobel 3×3 gradient (Gx, Gy).
//!   4. Gradient magnitude M = sqrt(Gx² + Gy²).
//!   5. Double threshold + 8-connected hysteresis.
//!   6. Morphological dilate (elliptical approx, r=7, 2 iters).
//!   7. Morphological close (elliptical approx, r=15, 1 iter).
//!   8. Flood-fill enclosed holes from border.
//!
//! Output: binary mask same size as input image (0 or 1), only set in
//! the padded bbox region.

use anyhow::Result;

use crate::domain::{BBox, PageKind};

// ── Tuning constants (match doc) ─────────────────────────────────────────

const EDGE_LOW:   f32 = 25.0;
const EDGE_HIGH:  f32 = 100.0;
const DILATE_R:   usize = 7;
const DILATE_ITER: usize = 2;
const CLOSE_R:    usize = 15;
const CLOSE_ITER: usize = 1;
const PAD_FRAC:   f32 = 0.15;
const PAD_MIN:    i32 = 8;

pub fn detect_strokes(
    img: &[u8],     // flat RGB, 3 bytes/pixel, W×H
    img_w: usize,
    img_h: usize,
    bbox: BBox,
    _page_kind: PageKind,   // reserved: future LAB ΔE for color
) -> Result<Vec<u8>> {
    // 1. Pad
    let short = bbox.short_edge() as f32;
    let pad   = ((short * PAD_FRAC).round() as i32).max(PAD_MIN);
    let crop  = bbox.expand(pad, img_w as i32, img_h as i32);
    let cw    = crop.width() as usize;
    let ch    = crop.height() as usize;
    let cx0   = crop.x1 as usize;
    let cy0   = crop.y1 as usize;

    // 2. Extract + RGB → gray
    let mut gray = vec![0f32; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let si = ((cy0 + y) * img_w + (cx0 + x)) * 3;
            let r = img[si] as f32;
            let g = img[si + 1] as f32;
            let b = img[si + 2] as f32;
            gray[y * cw + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }

    // 3-4. Sobel magnitude
    let mag = sobel_magnitude(&gray, cw, ch);

    // 5. Double threshold + hysteresis
    let edges = hysteresis(&mag, cw, ch, EDGE_LOW, EDGE_HIGH);

    // 6. Dilate
    let mut mask = edges;
    for _ in 0..DILATE_ITER {
        mask = dilate_ellipse(&mask, cw, ch, DILATE_R);
    }

    // 7. Close = dilate + erode
    for _ in 0..CLOSE_ITER {
        let d = dilate_ellipse(&mask, cw, ch, CLOSE_R);
        mask  = erode_ellipse(&d, cw, ch, CLOSE_R);
    }

    // 8. Fill enclosed holes
    fill_enclosed_holes(&mut mask, cw, ch);

    // Embed crop result back into full-page mask
    let mut out = vec![0u8; img_w * img_h];
    for y in 0..ch {
        for x in 0..cw {
            out[(cy0 + y) * img_w + (cx0 + x)] = mask[y * cw + x];
        }
    }
    Ok(out)
}

// ── Sobel magnitude ───────────────────────────────────────────────────────

fn sobel_magnitude(gray: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let tl = gray[(y-1)*w + (x-1)]; let tc = gray[(y-1)*w + x]; let tr = gray[(y-1)*w + (x+1)];
            let ml = gray[ y   *w + (x-1)];                               let mr = gray[ y   *w + (x+1)];
            let bl = gray[(y+1)*w + (x-1)]; let bc = gray[(y+1)*w + x]; let br = gray[(y+1)*w + (x+1)];
            let gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
            let gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
            mag[y*w + x] = (gx*gx + gy*gy).sqrt();
        }
    }
    mag
}

// ── Double threshold + 8-connected hysteresis ────────────────────────────

fn hysteresis(mag: &[f32], w: usize, h: usize, low: f32, high: f32) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    // Mark strong edges
    for i in 0..w * h {
        if mag[i] >= high { out[i] = 2; }
        else if mag[i] >= low { out[i] = 1; }
    }
    // Promote weak edges 8-connected to strong
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if out[i] != 1 { continue; }
                let has_strong = [
                    out[(y-1)*w + (x-1)], out[(y-1)*w + x], out[(y-1)*w + (x+1)],
                    out[ y   *w + (x-1)],                    out[ y   *w + (x+1)],
                    out[(y+1)*w + (x-1)], out[(y+1)*w + x], out[(y+1)*w + (x+1)],
                ].iter().any(|&v| v == 2);
                if has_strong { out[i] = 2; changed = true; }
            }
        }
    }
    // Finalise: 2 → 1, 1 → 0
    for v in out.iter_mut() { *v = if *v == 2 { 1 } else { 0 }; }
    out
}

// ── Elliptical approximation morphology ──────────────────────────────────

/// Build circular structuring element using distance check.
fn make_disk(r: usize) -> (Vec<(i32, i32)>, usize) {
    let r2 = (r * r) as f64;
    let ri = r as i32;
    let mut disk = Vec::new();
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if (dx*dx + dy*dy) as f64 <= r2 {
                disk.push((dx, dy));
            }
        }
    }
    let diam = 2 * r + 1;
    (disk, diam)
}

fn dilate_ellipse(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let (disk, _) = make_disk(r);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            for &(dx, dy) in &disk {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    if src[(ny as usize) * w + (nx as usize)] != 0 {
                        out[y * w + x] = 1;
                        break;
                    }
                }
            }
        }
    }
    out
}

fn erode_ellipse(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let (disk, _) = make_disk(r);
    let mut out = vec![1u8; w * h];
    for y in 0..h {
        for x in 0..w {
            for &(dx, dy) in &disk {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let outside = nx < 0 || ny < 0 || (nx as usize) >= w || (ny as usize) >= h;
                if outside || src[(ny as usize) * w + (nx as usize)] == 0 {
                    out[y * w + x] = 0;
                    break;
                }
            }
        }
    }
    out
}

// ── Flood-fill enclosed holes ─────────────────────────────────────────────
// BFS from all border=0 pixels → mark reachable as "outside".
// Remaining 0 pixels unreachable from border → flip to 1.

pub fn fill_enclosed_holes(mask: &mut Vec<u8>, w: usize, h: usize) {
    if w == 0 || h == 0 { return; }
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();

    let push = |idx: usize, mask: &[u8], outside: &mut Vec<bool>, stack: &mut Vec<usize>| {
        if mask[idx] == 0 && !outside[idx] {
            outside[idx] = true;
            stack.push(idx);
        }
    };

    for x in 0..w { push(x, mask, &mut outside, &mut stack); push((h-1)*w+x, mask, &mut outside, &mut stack); }
    for y in 0..h { push(y*w, mask, &mut outside, &mut stack); push(y*w+(w-1), mask, &mut outside, &mut stack); }

    while let Some(i) = stack.pop() {
        let x = i % w; let y = i / w;
        if x > 0     { let n = i-1; if mask[n]==0 && !outside[n] { outside[n]=true; stack.push(n); } }
        if x+1 < w   { let n = i+1; if mask[n]==0 && !outside[n] { outside[n]=true; stack.push(n); } }
        if y > 0     { let n = i-w; if mask[n]==0 && !outside[n] { outside[n]=true; stack.push(n); } }
        if y+1 < h   { let n = i+w; if mask[n]==0 && !outside[n] { outside[n]=true; stack.push(n); } }
    }

    for i in 0..w*h {
        if mask[i] == 0 && !outside[i] { mask[i] = 1; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hole_fill_closed_ring() {
        let mut m = vec![0u8; 25]; // 5×5
        for x in 1..4 { m[5+x]=1; m[15+x]=1; }
        for y in 1..4 { m[y*5+1]=1; m[y*5+3]=1; }
        fill_enclosed_holes(&mut m, 5, 5);
        assert_eq!(m[12], 1); // centre filled
        assert_eq!(m[0],  0); // corner stays 0
    }

    #[test]
    fn sobel_detects_vertical_edge() {
        // 6×6 gray: left half=0, right half=255
        let mut gray = vec![0f32; 36];
        for y in 0..6 { for x in 3..6 { gray[y*6+x] = 255.0; } }
        let mag = sobel_magnitude(&gray, 6, 6);
        // column x=2 should have strong gradient
        assert!(mag[1*6+2] > 100.0);
    }
}
