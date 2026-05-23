// SPDX-License-Identifier: GPL-3.0-or-later
//! Route decision per inpaint region.

use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};

use crate::adapters::flat_fill;
use crate::domain::{BBox, PadProfile};
use crate::pipeline::regions::Region;

const SNAP_MOD:             usize = 8;
const DEFAULT_AOT_CANVAS:   usize = 512;
const TARGET_MASK_DENSITY:  f32   = 0.42;
const MIN_CONTEXT_PX:       i32   = 32;

pub enum Route {
    FlatFill { color: [u8; 3] },
    Aot      { tile: AotTile  },
}

pub struct AotTile {
    pub rgb:      Vec<u8>,
    pub mask:     Vec<u8>,
    pub canvas_w: usize,
    pub canvas_h: usize,
    pub pad_box:  BBox,
}

pub fn decide(
    region:  &Region,
    rgb:     &[u8],
    mask:    &[u8],
    img_w:   usize,
    img_h:   usize,
    profile: &PadProfile,
) -> Route {
    // Try flat fill first (fast path — skip AOT entirely)
    if let Some(color) = flat_fill::probe(rgb, mask, img_w, img_h, &region.bbox) {
        return Route::FlatFill { color };
    }
    Route::Aot { tile: build_tile(rgb, mask, img_w, img_h, &region.bbox, profile) }
}

// ── AOT tile builder ─────────────────────────────────────────────────────

fn build_tile(
    rgb:   &[u8],
    mask:  &[u8],
    src_w: usize,
    src_h: usize,
    bb:    &BBox,
    profile: &PadProfile,
) -> AotTile {
    let canvas = canvas_size();
    let bw = bb.width() as f32;
    let bh = bb.height() as f32;
    let short = bw.min(bh);
    let mut context = ((short * profile.context_frac).round() as i32).max(MIN_CONTEXT_PX);
    let mut crop = expand_crop(bb, context, src_w as i32, src_h as i32);

    // Grow context until mask density ≤ TARGET_MASK_DENSITY or full page
    for _ in 0..16 {
        let mask_on = count_mask(mask, src_w, &crop);
        let density = mask_on as f32 / (canvas * canvas) as f32;
        let full_page = crop.x1 == 0 && crop.y1 == 0
            && crop.x2 == src_w as i32 && crop.y2 == src_h as i32;
        if density <= TARGET_MASK_DENSITY || full_page { break; }
        context += (context / 2).max(16);
        crop = expand_crop(bb, context, src_w as i32, src_h as i32);
    }

    let cw = crop.width()  as usize;
    let ch = crop.height() as usize;
    let x0 = crop.x1 as usize;
    let y0 = crop.y1 as usize;

    let mut crop_rgb  = vec![0u8; cw * ch * 3];
    let mut crop_mask = vec![0u8; cw * ch];
    for ty in 0..ch {
        for tx in 0..cw {
            let si = (y0 + ty) * src_w + (x0 + tx);
            let di = ty * cw + tx;
            crop_rgb[di*3  ] = rgb[si*3  ];
            crop_rgb[di*3+1] = rgb[si*3+1];
            crop_rgb[di*3+2] = rgb[si*3+2];
            crop_mask[di]    = mask[si];
        }
    }

    let rgb_canvas  = resize_rgb(&crop_rgb,  cw, ch, canvas, canvas, FilterType::Triangle);
    let mask_canvas = resize_mask(&crop_mask, cw, ch, canvas, canvas);

    AotTile {
        rgb:      rgb_canvas,
        mask:     mask_canvas,
        canvas_w: canvas,
        canvas_h: canvas,
        pad_box:  crop,
    }
}

fn canvas_size() -> usize {
    std::env::var("AOT_CANVAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= SNAP_MOD)
        .map(|v| snap_up(v, SNAP_MOD))
        .unwrap_or(DEFAULT_AOT_CANVAS)
}

fn snap_up(v: usize, m: usize) -> usize { ((v + m - 1) / m) * m }

fn expand_crop(bb: &BBox, ctx: i32, w: i32, h: i32) -> BBox {
    BBox::new(
        (bb.x1 - ctx).max(0),
        (bb.y1 - ctx).max(0),
        (bb.x2 + ctx).min(w),
        (bb.y2 + ctx).min(h),
    ).unwrap_or(*bb)
}

fn count_mask(mask: &[u8], src_w: usize, bb: &BBox) -> usize {
    let mut n = 0;
    for y in bb.y1..bb.y2 {
        for x in bb.x1..bb.x2 {
            if mask[y as usize * src_w + x as usize] >= 127 { n += 1; }
        }
    }
    n
}

fn resize_rgb(src: &[u8], w: usize, h: usize, ow: usize, oh: usize, f: FilterType) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, src.to_vec()).expect("rgb shape");
    image::imageops::resize(&img, ow as u32, oh as u32, f).into_raw()
}

fn resize_mask(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let img: ImageBuffer<image::Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, src.to_vec()).expect("mask shape");
    image::imageops::resize(&img, ow as u32, oh as u32, FilterType::Nearest)
        .into_raw()
        .into_iter()
        .map(|v| if v >= 127 { 255 } else { 0 })
        .collect()
}
