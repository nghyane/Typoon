//! Pure-Rust render core, shared by the Python (`pyo3`) and WASM
//! (`wasm-bindgen`) entry points.
//!
//! Wire shape is canonical RGBA `[H, W, 4]` u8 from end to end:
//!   - input is the inpainted page (text already erased)
//!   - the core mutates that buffer in place to draw translated text
//!   - output is the same RGBA layout, ready for codec encode
//!
//! No "original" page is needed — the inpaint pass only erases text
//! pixels; bubble outlines stay intact in the clean canvas.
//!
//! No FFI types, no `numpy`, no `js_sys`. Each entry-point module
//! translates its native call shape into these structs and back.

use crate::{fit, layout, overlay, shape};

/// Per-bubble source-font hint, fed to the layout fitter so target
/// translations follow the source's visual rhythm.
#[derive(Clone, Debug)]
pub struct TypesettingHint {
    pub font_size_px:       u32,
    pub line_count:         u32,
    pub avg_chars_per_line: f64,
    /// "vertical" | "horizontal" — source script direction. "vertical"
    /// skips refinement (source font sizing is meaningless for horizontal
    /// Vietnamese target).
    pub text_direction:     String,
}

/// Per-bubble layout result reported back to the caller.
#[derive(Clone, Debug)]
pub struct BubbleInfo {
    pub font_size_px: u32,
    pub line_height:  f64,
    pub overflow:     bool,
    pub rect:         [f64; 4],
}

/// Output of `render_page`: rendered RGBA pixels + per-bubble metrics.
///
/// RGBA (not RGB) because:
///   - upstream codecs (@jsquash) work in RGBA natively, so RGB↔RGBA
///     conversions on both sides were costing 11 MB×2 per page and the
///     conversion loops dominated wall-time on 1600×2400 pages
///   - tiny-skia + image::RgbaImage are RGBA-native too, so the render
///     core never has to leave that representation internally
///   - PNG output is RGBA — the JS side passes the buffer straight to
///     @jsquash/png encode with zero conversion
pub struct RenderedPage {
    pub width:   u32,
    pub height:  u32,
    pub rgba:    Vec<u8>,
    pub bubbles: Vec<BubbleInfo>,
}

/// Pure render entry point (RGBA-native, single-buffer).
///
/// `clean_rgba` is taken by value and mutated in place. Caller must
/// not hold a second reference.
///
/// `polygons[i]` is the **drawable** region for bubble `i` — already
/// the inner text area as decided by the grouper (Lens word union +
/// padding, clipped to the page). No border scanning, no bubble-mask
/// expansion: the polygon IS the area we paint into.
pub fn render_page_rgba(
    clean_rgba:    Vec<u8>,
    width:         u32,
    height:        u32,
    polygons:      Vec<Vec<[f64; 2]>>,
    texts:         Vec<String>,
    page_width:    u32,
    hints:         Vec<Option<TypesettingHint>>,
) -> anyhow::Result<RenderedPage> {
    if polygons.len() != texts.len() {
        anyhow::bail!("polygons and texts must have same length");
    }
    let pixel_count = (width as usize) * (height as usize);
    if clean_rgba.len() != pixel_count * 4 {
        anyhow::bail!("clean_rgba: expected {} bytes, got {}", pixel_count * 4, clean_rgba.len());
    }

    let hints_vec: Vec<Option<TypesettingHint>> = if hints.is_empty() {
        vec![None; polygons.len()]
    } else {
        if hints.len() != polygons.len() {
            anyhow::bail!("hints length must match polygons");
        }
        hints
    };

    // Single owned canvas — used as the render target only. Borders
    // are no longer scanned; the grouper hands us the drawable polygon.
    let canvas = image::RgbaImage::from_raw(width, height, clean_rgba)
        .ok_or_else(|| anyhow::anyhow!("clean_rgba: invalid {}x{} buffer", width, height))?;

    let bubbles: Vec<(overlay::RenderBubble, Option<TypesettingHint>)> = polygons
        .iter()
        .zip(&texts)
        .zip(hints_vec.into_iter())
        .filter(|((_, text), _)| !text.is_empty())
        .map(|((poly, text), hint)| {
            let area = layout::DrawableArea::from_polygon(poly, layout::DEFAULT_INSET);
            (
                overlay::RenderBubble {
                    translated_text: text.clone(),
                    area,
                    font_size_px: 0,
                    line_height:  0.0,
                },
                hint,
            )
        })
        .collect();

    let fit_items: Vec<(&str, &layout::DrawableArea, Option<fit::FitHint>)> = bubbles
        .iter()
        .map(|(b, h)| {
            let fit_hint = h.as_ref().map(|h| fit::FitHint {
                font_size_px:       h.font_size_px,
                line_count:         h.line_count,
                avg_chars_per_line: h.avg_chars_per_line,
                text_direction:     if h.text_direction == "vertical" {
                    fit::TextDirection::Vertical
                } else {
                    fit::TextDirection::Horizontal
                },
            });
            (b.translated_text.as_str(), &b.area, fit_hint)
        })
        .collect();

    let fits = fit::FitEngine::fit_page_areas(&fit_items, page_width)?;

    let fitted: Vec<overlay::RenderBubble> = bubbles
        .into_iter()
        .zip(&fits)
        .map(|((mut b, _), f)| {
            b.translated_text = f.text.clone();
            b.font_size_px    = f.font_size_px;
            b.line_height     = f.line_height;
            b
        })
        .collect();

    let info: Vec<BubbleInfo> = fitted
        .iter()
        .zip(&fits)
        .map(|(b, f)| BubbleInfo {
            font_size_px: f.font_size_px,
            line_height:  f.line_height,
            overflow:     f.overflow,
            rect: {
                let (x, y, w, h) = b.area.rect();
                [x, y, w, h]
            },
        })
        .collect();

    // Render mutates `canvas` in place and returns it.
    let rendered = overlay::render(canvas, &fitted);
    let (w, h) = rendered.dimensions();
    let rgba_buf = rendered.into_raw();

    Ok(RenderedPage { width: w, height: h, rgba: rgba_buf, bubbles: info })
}

/// Compute character budget for a bubble: how many Vietnamese characters
/// fit at the font size the renderer will actually choose.
///
/// Returns `(chars_per_line, n_lines, font_size_px)`.
pub fn char_budget(
    bubble_w:         u32,
    bubble_h:         u32,
    page_width:       u32,
    src_font_size_px: u32,
    src_line_count:   u32,
) -> (u32, u32, u32) {
    if bubble_w == 0 || bubble_h == 0 {
        return (0, 0, 0);
    }

    let polygon = [
        [0.0_f64, 0.0],
        [bubble_w as f64, 0.0],
        [bubble_w as f64, bubble_h as f64],
        [0.0, bubble_h as f64],
    ];
    let area = layout::DrawableArea::from_polygon(&polygon, layout::DEFAULT_INSET);
    let (safe_w, safe_h) = area.size();
    if safe_w < 1.0 || safe_h < 1.0 {
        return (0, 0, 0);
    }

    // Hint reserved for future binary-search path; kept for parity with the
    // earlier PyO3-only implementation in case downstream code feeds priors.
    let _hint = if src_font_size_px > 0 && src_line_count > 0 {
        Some(crate::fit::FitHint {
            font_size_px:       src_font_size_px,
            line_count:         src_line_count,
            avg_chars_per_line: 20.0,
            text_direction:     crate::fit::TextDirection::Horizontal,
        })
    } else {
        None
    };

    let max_font = {
        let scaled = (page_width as f64 * 0.05) as u32;
        scaled.clamp(48, 96)
    };
    let line_count = if src_line_count > 0 {
        src_line_count
    } else {
        let fallback_font = (safe_h / 3.0).clamp(8.0, 32.0);
        (safe_h / (fallback_font * layout::LINE_HEIGHT_MULTIPLIER))
            .ceil()
            .max(1.0) as u32
    };
    let font_by_height = safe_h
        / (line_count as f64 * layout::LINE_HEIGHT_MULTIPLIER
            - (layout::LINE_HEIGHT_MULTIPLIER - 1.0));
    let font_px = (font_by_height as u32)
        .min(max_font)
        .min((safe_w * 0.85) as u32)
        .max(8);

    let line_h = font_px as f64 * layout::LINE_HEIGHT_MULTIPLIER;
    let tolerant = safe_h * (1.0 + 0.08);
    let n_lines = ((tolerant - font_px as f64) / line_h + 1.0)
        .floor()
        .max(1.0) as u32;

    let measure_str = "abcdefghijklmnopqrstuvwxyz àáảãạ êếểễệ ôốổỗộ ưứửữự";
    let str_w = shape::measure_width(measure_str, font_px);
    let n_chars = measure_str.chars().filter(|c| !c.is_whitespace()).count();
    let avg_glyph_w = if n_chars > 0 {
        str_w / n_chars as f64
    } else {
        font_px as f64 * 0.6
    };
    let chars_per_line = ((safe_w / avg_glyph_w) * 1.25).floor().max(1.0) as u32;

    (chars_per_line, n_lines, font_px)
}

// ─── Pure stitch core ───────────────────────────────────────────────────

/// One page input to `stitch_pages`: interleaved RGB `[H, W, 3]` u8.
pub struct StitchPage<'a> {
    pub rgb:    &'a [u8],
    pub width:  u32,
    pub height: u32,
}

pub struct StitchedStrip {
    pub rgb:          Vec<u8>,
    pub width:        u32,
    pub heights:      Vec<u32>,
    pub target_width: u32,
}

/// Stitch pages into one vertical strip with median-width normalisation.
/// Mirrors `crate::stitch::stitch_pages` but without numpy indirection.
pub fn stitch_pages(pages: &[StitchPage<'_>]) -> anyhow::Result<StitchedStrip> {
    if pages.is_empty() {
        anyhow::bail!("no pages");
    }
    for (i, p) in pages.iter().enumerate() {
        let expected = (p.width as usize) * (p.height as usize) * 3;
        if p.rgb.len() != expected {
            anyhow::bail!("page[{}]: rgb {} != {}x{}x3", i, p.rgb.len(), p.width, p.height);
        }
    }

    if pages.len() == 1 {
        let p = &pages[0];
        return Ok(StitchedStrip {
            rgb:          p.rgb.to_vec(),
            width:        p.width,
            heights:      vec![p.height],
            target_width: p.width,
        });
    }

    let mut widths: Vec<usize> = pages.iter().map(|p| p.width as usize).collect();
    widths.sort_unstable();
    let target_w = widths[widths.len() / 2];

    let out_heights: Vec<usize> = pages
        .iter()
        .map(|p| {
            let (h, w) = (p.height as usize, p.width as usize);
            if w > target_w {
                ((h as f64) * (target_w as f64) / (w as f64)).round() as usize
            } else {
                h
            }
        })
        .collect();

    let total_h: usize = out_heights.iter().sum();
    let mut buf = vec![255u8; total_h * target_w * 3];

    let mut y_off: usize = 0;
    for (i, page) in pages.iter().enumerate() {
        let src = page.rgb;
        let src_w = page.width as usize;
        let src_h = page.height as usize;
        let dst_h = out_heights[i];

        if src_w == target_w {
            for row in 0..src_h.min(dst_h) {
                let s = row * src_w * 3;
                let d = (y_off + row) * target_w * 3;
                let n = src_w * 3;
                buf[d..d + n].copy_from_slice(&src[s..s + n]);
            }
        } else if src_w > target_w {
            let sx = src_w as f64 / target_w as f64;
            let sy = src_h as f64 / dst_h as f64;
            for dy in 0..dst_h {
                let fy = (dy as f64 * sy).min((src_h - 1) as f64);
                let y0 = fy as usize;
                let y1 = (y0 + 1).min(src_h - 1);
                let wy = fy - y0 as f64;
                for dx in 0..target_w {
                    let fx = (dx as f64 * sx).min((src_w - 1) as f64);
                    let x0 = fx as usize;
                    let x1 = (x0 + 1).min(src_w - 1);
                    let wx = fx - x0 as f64;
                    let di = ((y_off + dy) * target_w + dx) * 3;
                    for c in 0..3 {
                        let v00 = src[(y0 * src_w + x0) * 3 + c] as f64;
                        let v01 = src[(y0 * src_w + x1) * 3 + c] as f64;
                        let v10 = src[(y1 * src_w + x0) * 3 + c] as f64;
                        let v11 = src[(y1 * src_w + x1) * 3 + c] as f64;
                        let v = v00 * (1.0 - wx) * (1.0 - wy)
                            + v01 * wx * (1.0 - wy)
                            + v10 * (1.0 - wx) * wy
                            + v11 * wx * wy;
                        buf[di + c] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        } else {
            for row in 0..src_h {
                let s = row * src_w * 3;
                let d = (y_off + row) * target_w * 3;
                let n = src_w * 3;
                buf[d..d + n].copy_from_slice(&src[s..s + n]);
            }
        }

        y_off += dst_h;
    }

    Ok(StitchedStrip {
        rgb:          buf,
        width:        target_w as u32,
        heights:      out_heights.iter().map(|&h| h as u32).collect(),
        target_width: target_w as u32,
    })
}
