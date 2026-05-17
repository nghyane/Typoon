//! wasm-bindgen entry (Cloudflare Workers typeset stage).
//!
//! Wire shape — keeps the host-side JS contract narrow:
//!
//! ```ts
//! // worker side
//! import init, { render_page, char_budget, stitch_pages, version } from "./typoon_render.js";
//! await init(wasmModule);                  // CompiledWasm import
//!
//! const out: RenderedPageJs = render_page({
//!   original_rgb: Uint8Array,              // [H, W, 3] interleaved
//!   clean_rgb:    Uint8Array,
//!   width:  number, height: number,
//!   polygons:   number[][][],              // [[ [x,y], ... ], ...]
//!   texts:      string[],
//!   page_width: number,
//!   hints?:     Array<Hint | null>,
//! });
//! // out.rgb is RGB [H, W, 3]; PNG encode happens JS-side via @jsquash/png.
//! ```
//!
//! No PNG/JPEG codec is shipped from Rust — the worker bundles `@jsquash/png`
//! once and reuses it across the inpaint + typeset stages, saving ~200 KiB
//! of WASM duplication.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::core;

/// Called once when the WASM module is loaded. Routes panics through
/// `console.error` instead of the cryptic `RuntimeError: unreachable`.
#[wasm_bindgen(start)]
pub fn __start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Set a property on a JS object, propagating any error as a `JsError`.
fn set(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsError> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|e| {
            JsError::new(&format!(
                "Reflect::set({key}) failed: {}",
                e.as_string().unwrap_or_else(|| "<no message>".into()),
            ))
        })
}

// ─── JSON-side DTOs ────────────────────────────────────────────────────────
//
// `serde-wasm-bindgen` is the canonical bridge for structured JS↔Rust value
// passing. Buffers (`original_rgb`, `clean_rgb`) come in as `Uint8Array` to
// avoid a double copy through JSON parse.

#[derive(Deserialize)]
struct HintIn {
    font_size_px:       u32,
    line_count:         u32,
    avg_chars_per_line: f64,
    #[serde(default = "default_horizontal")]
    text_direction:     String,
}

fn default_horizontal() -> String { "horizontal".into() }

impl From<HintIn> for core::TypesettingHint {
    fn from(h: HintIn) -> Self {
        Self {
            font_size_px:       h.font_size_px,
            line_count:         h.line_count,
            avg_chars_per_line: h.avg_chars_per_line,
            text_direction:     h.text_direction,
        }
    }
}

#[derive(Deserialize)]
struct RenderRequest {
    width:        u32,
    height:       u32,
    polygons:     Vec<Vec<[f64; 2]>>,
    texts:        Vec<String>,
    page_width:   u32,
    #[serde(default)]
    hints:        Vec<Option<HintIn>>,
}

#[derive(Serialize)]
struct BubbleOut {
    font_size_px: u32,
    line_height:  f64,
    overflow:     bool,
    rect:         [f64; 4],
}

impl From<core::BubbleInfo> for BubbleOut {
    fn from(b: core::BubbleInfo) -> Self {
        Self {
            font_size_px: b.font_size_px,
            line_height:  b.line_height,
            overflow:     b.overflow,
            rect:         b.rect,
        }
    }
}

/// Render a page (RGBA-native, single-canvas).
///
/// Input is the inpaint output (clean page) as RGBA `[H, W, 4]`. The
/// `polygons` array carries the **drawable** region per bubble — already
/// the inner text area as decided by the grouper. No border scanning.
///
/// Output is `{ width, height, rgba: Uint8Array, bubbles: [...] }`.
/// JS encodes the returned RGBA straight via `@jsquash/png`.
#[wasm_bindgen]
pub fn render_page(
    request:        JsValue,
    clean_rgba:     Vec<u8>,
) -> Result<JsValue, JsError> {
    let req: RenderRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|e| JsError::new(&format!("invalid request: {e}")))?;

    let hints: Vec<Option<core::TypesettingHint>> = req
        .hints
        .into_iter()
        .map(|h| h.map(Into::into))
        .collect();

    let out = core::render_page_rgba(
        clean_rgba,
        req.width,
        req.height,
        req.polygons,
        req.texts,
        req.page_width,
        hints,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let obj = js_sys::Object::new();
    set(&obj, "width",  &JsValue::from(out.width))?;
    set(&obj, "height", &JsValue::from(out.height))?;
    let rgba_array = js_sys::Uint8Array::new_with_length(out.rgba.len() as u32);
    rgba_array.copy_from(&out.rgba);
    set(&obj, "rgba", &rgba_array)?;
    let bubbles_out: Vec<BubbleOut> = out.bubbles.into_iter().map(Into::into).collect();
    let bubbles = serde_wasm_bindgen::to_value(&bubbles_out)
        .map_err(|e| JsError::new(&format!("serialize bubbles: {e}")))?;
    set(&obj, "bubbles", &bubbles)?;
    Ok(obj.into())
}

/// Character budget for a bubble. Returns `[chars_per_line, n_lines, font_size_px]`.
#[wasm_bindgen]
pub fn char_budget(
    bubble_w:         u32,
    bubble_h:         u32,
    page_width:       u32,
    src_font_size_px: u32,
    src_line_count:   u32,
) -> Box<[u32]> {
    let (cpl, nl, fp) = core::char_budget(
        bubble_w, bubble_h, page_width, src_font_size_px, src_line_count,
    );
    Box::new([cpl, nl, fp])
}

// ─── Stitch ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct StitchPageIn {
    width:  u32,
    height: u32,
}

#[derive(Deserialize)]
struct StitchRequest {
    pages: Vec<StitchPageIn>,
}

/// Stitch RGB pages into a vertical strip.
///
/// `pages_meta` is `{ pages: [{ width, height }, ...] }`. `pages_data` is
/// the concatenation of each page's RGB buffer in declaration order; the
/// helper sums `page.width * page.height * 3` to know where each page
/// starts.
#[wasm_bindgen]
pub fn stitch_pages(
    pages_meta: JsValue,
    pages_data: &[u8],
) -> Result<JsValue, JsError> {
    let req: StitchRequest = serde_wasm_bindgen::from_value(pages_meta)
        .map_err(|e| JsError::new(&format!("invalid request: {e}")))?;

    let mut offsets: Vec<usize> = Vec::with_capacity(req.pages.len() + 1);
    offsets.push(0);
    let mut acc = 0usize;
    for p in &req.pages {
        acc = acc.saturating_add((p.width as usize) * (p.height as usize) * 3);
        offsets.push(acc);
    }
    if acc != pages_data.len() {
        return Err(JsError::new(&format!(
            "pages_data length {} != sum of W*H*3 = {}",
            pages_data.len(), acc,
        )));
    }

    let pages: Vec<core::StitchPage<'_>> = req
        .pages
        .iter()
        .enumerate()
        .map(|(i, p)| core::StitchPage {
            rgb:    &pages_data[offsets[i]..offsets[i + 1]],
            width:  p.width,
            height: p.height,
        })
        .collect();

    let out = core::stitch_pages(&pages).map_err(|e| JsError::new(&e.to_string()))?;

    let obj = js_sys::Object::new();
    set(&obj, "width",        &JsValue::from(out.width))?;
    set(&obj, "target_width", &JsValue::from(out.target_width))?;
    let heights = js_sys::Array::new();
    for h in &out.heights {
        heights.push(&JsValue::from(*h));
    }
    set(&obj, "heights", &heights)?;
    let rgb_array = js_sys::Uint8Array::new_with_length(out.rgb.len() as u32);
    rgb_array.copy_from(&out.rgb);
    set(&obj, "rgb", &rgb_array)?;
    Ok(obj.into())
}
