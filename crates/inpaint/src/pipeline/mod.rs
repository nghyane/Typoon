// SPDX-License-Identifier: GPL-3.0-or-later
//! Inpaint pipeline: pure orchestration, no I/O.

mod close;
mod compose;
mod regions;
mod route;
pub mod sources;

pub use regions::Region;

use anyhow::Result;
use std::path::Path;

use crate::adapters::png_codec;
use crate::domain::InpaintPlan;
use crate::Inpainter;

/// Debug-dump sink — writes intermediate buffers if `INPAINT_DEBUG_DIR` is set.
pub struct DebugSink(Option<std::path::PathBuf>);

impl DebugSink {
    pub fn from_env() -> Self {
        Self(std::env::var_os("INPAINT_DEBUG_DIR").map(Into::into))
    }
    pub fn from_path(p: Option<&Path>) -> Self {
        Self(p.map(|x| x.to_path_buf()))
    }
    pub fn mask(&self, name: &str, data: &[u8], w: usize, h: usize) {
        let Some(ref dir) = self.0 else { return };
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{name}.png"));
        if let Some(buf) = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(w as u32, h as u32, data) {
            let _ = buf.save(path);
        }
    }
    pub fn rgb(&self, name: &str, data: &[u8], w: usize, h: usize) {
        let Some(ref dir) = self.0 else { return };
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{name}.png"));
        if let Some(buf) = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, data) {
            let _ = buf.save(path);
        }
    }
}

// ── Public entry ─────────────────────────────────────────────────────────

/// Process one page synchronously. Returns PNG bytes.
///
/// `img_bytes` — JPEG-encoded prepared page.
/// `plan_bytes` — msgpack-encoded `InpaintPlan`.
/// `debug_dir`  — optional path; when `Some`, intermediate images are
///                written there for visual verification.
pub fn run_page(
    inpainter:  &Inpainter,
    img_bytes:  Vec<u8>,
    plan_bytes: Vec<u8>,
    debug_dir:  Option<&Path>,
) -> Result<Vec<u8>> {
    let dbg = DebugSink::from_path(debug_dir);

    let (rgb, w, h) = png_codec::decode_jpeg(&img_bytes)?;
    let plan        = InpaintPlan::from_msgpack(&plan_bytes)?;

    if plan.page_w() as usize != w || plan.page_h() as usize != h {
        anyhow::bail!(
            "plan size {}×{} != image {}×{}",
            plan.page_w(), plan.page_h(), w, h
        );
    }

    dbg.rgb("00_input", &rgb, w, h);

    // 1. Build per-group masks → OR into page mask
    let page_mask = sources::build_page_mask(&rgb, w, h, &plan, &dbg)?;
    dbg.mask("01_page_mask", &page_mask, w, h);

    // 2. Merge groups → inpaint regions
    let region_list = regions::merge(&plan.groups);

    let mut composite = rgb;
    dbg.rgb("02_pre_inpaint", &composite, w, h);

    if region_list.is_empty() {
        return png_codec::encode_png(&composite, w, h);
    }

    // 3. Route each region: flat-fill or AOT
    for (ri, region) in region_list.iter().enumerate() {
        let decision = route::decide(region, &composite, &page_mask, w, h);

        match decision {
            route::Route::FlatFill { color } => {
                tracing::info!(region = ri, ?color, "flat fill");
                compose::fill_flat(&mut composite, &page_mask, w, &region.bbox, color);
            }
            route::Route::Aot { tile } => {
                tracing::info!(
                    region = ri,
                    tile = %format!("{}×{}", tile.canvas_w, tile.canvas_h),
                    "aot inpaint"
                );
                dbg.rgb(&format!("tile_{ri:02}_rgb"),  &tile.rgb,  tile.canvas_w, tile.canvas_h);
                dbg.mask(&format!("tile_{ri:02}_mask"), &tile.mask, tile.canvas_w, tile.canvas_h);

                let out = inpainter.inpaint(
                    &tile.rgb, &tile.mask,
                    tile.canvas_w as u32, tile.canvas_h as u32,
                )?;

                dbg.rgb(&format!("tile_{ri:02}_out"), &out, tile.canvas_w, tile.canvas_h);
                compose::paste_tile(&mut composite, &page_mask, w, h, &tile.pad_box, tile.canvas_w, tile.canvas_h, &out);
            }
        }
    }

    dbg.rgb("99_final", &composite, w, h);
    png_codec::encode_png(&composite, w, h)
}
