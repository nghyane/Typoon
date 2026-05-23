// SPDX-License-Identifier: GPL-3.0-or-later
//! Compose: paste inpainted output back onto composite RGB.

use crate::domain::BBox;
use image::{ImageBuffer, Rgb};
use image::imageops::FilterType;

const FEATHER_RADIUS: i32 = 3;

// ── Flat fill ─────────────────────────────────────────────────────────────

pub fn fill_flat(
    composite: &mut [u8],
    mask:      &[u8],
    w:         usize,
    bb:        &BBox,
    color:     [u8; 3],
) {
    let h = mask.len() / w;
    let x0 = bb.x1.max(0) as usize;
    let y0 = bb.y1.max(0) as usize;
    let x1 = bb.x2.min(w as i32 - 1) as usize;
    let y1 = bb.y2.min(h as i32 - 1) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let mi = y * w + x;
            if mask[mi] < 127 { continue; }
            let alpha = feather(mask, w, x, y, FEATHER_RADIUS);
            let p = mi * 3;
            for c in 0..3 {
                let old = composite[p + c] as f32;
                let new = color[c] as f32;
                composite[p + c] = (old * (1.0 - alpha) + new * alpha).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ── AOT tile paste ────────────────────────────────────────────────────────

pub fn paste_tile(
    composite: &mut [u8],
    orig_mask: &[u8],
    src_w:     usize,
    src_h:     usize,
    pad_box:   &BBox,
    tile_w:    usize,
    tile_h:    usize,
    tile_rgb:  &[u8],
) {
    let w0 = pad_box.width()  as usize;
    let h0 = pad_box.height() as usize;
    // Downscale tile back to crop size
    let crop_out = resize_rgb(tile_rgb, tile_w, tile_h, w0, h0);
    let x0 = pad_box.x1 as usize;
    let y0 = pad_box.y1 as usize;
    for ty in 0..h0 {
        let dst_y = y0 + ty;
        if dst_y >= src_h { continue; }
        for tx in 0..w0 {
            let dst_x = x0 + tx;
            if dst_x >= src_w { continue; }
            let mi = dst_y * src_w + dst_x;
            if orig_mask[mi] < 127 { continue; }
            let alpha = feather(orig_mask, src_w, dst_x, dst_y, FEATHER_RADIUS);
            let di = mi * 3;
            let ti = (ty * w0 + tx) * 3;
            for c in 0..3 {
                let old = composite[di + c] as f32;
                let new = crop_out[ti + c] as f32;
                composite[di + c] = (old * (1.0 - alpha) + new * alpha).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ── Feather alpha ─────────────────────────────────────────────────────────

fn feather(mask: &[u8], w: usize, x: usize, y: usize, radius: i32) -> f32 {
    let h = mask.len() / w;
    let mut min_dist = radius + 1;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d = dx.abs() + dy.abs();
            if d == 0 || d > radius { continue; }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32
               || mask[ny as usize * w + nx as usize] < 127 {
                min_dist = min_dist.min(d);
            }
        }
    }
    if min_dist > radius { 1.0 } else { min_dist as f32 / (radius + 1) as f32 }
}

fn resize_rgb(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, src.to_vec()).expect("rgb shape");
    image::imageops::resize(&img, ow as u32, oh as u32, FilterType::Triangle).into_raw()
}
