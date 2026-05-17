//! PyO3 extension entry (Python-only).
//!
//! Wraps `crate::core` in numpy-aware classes that match the existing
//! `typoon_render` module surface in Python.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::core;

/// Per-bubble typesetting hint derived from the source detector.
#[pyclass]
#[derive(Clone, Debug)]
pub struct TypesettingHint {
    #[pyo3(get, set)] pub font_size_px:       u32,
    #[pyo3(get, set)] pub line_count:         u32,
    #[pyo3(get, set)] pub avg_chars_per_line: f64,
    #[pyo3(get, set)] pub text_direction:     String,
}

#[pymethods]
impl TypesettingHint {
    #[new]
    #[pyo3(signature = (font_size_px, line_count, avg_chars_per_line, text_direction="horizontal"))]
    fn new(font_size_px: u32, line_count: u32, avg_chars_per_line: f64, text_direction: &str) -> Self {
        Self {
            font_size_px,
            line_count,
            avg_chars_per_line,
            text_direction: text_direction.to_string(),
        }
    }
}

impl From<TypesettingHint> for core::TypesettingHint {
    fn from(h: TypesettingHint) -> Self {
        Self {
            font_size_px:       h.font_size_px,
            line_count:         h.line_count,
            avg_chars_per_line: h.avg_chars_per_line,
            text_direction:     h.text_direction,
        }
    }
}

#[pyclass]
struct RenderResult {
    #[pyo3(get)] image:   Py<PyArray3<u8>>,
    #[pyo3(get)] bubbles: Vec<BubbleInfo>,
}

#[pyclass]
#[derive(Clone)]
struct BubbleInfo {
    #[pyo3(get)] font_size_px: u32,
    #[pyo3(get)] line_height:  f64,
    #[pyo3(get)] overflow:     bool,
    #[pyo3(get)] rect:         [f64; 4],
}

impl From<core::BubbleInfo> for BubbleInfo {
    fn from(b: core::BubbleInfo) -> Self {
        Self {
            font_size_px: b.font_size_px,
            line_height:  b.line_height,
            overflow:     b.overflow,
            rect:         b.rect,
        }
    }
}

/// Render translated text onto an inpainted page.
///
/// Wire shape (single-canvas, RGBA-only):
///
///   typoon_render.render(
///       clean: np.ndarray[uint8],          # [H, W, 4] inpaint output
///       polygons: list[list[[x, y]]],      # already the drawable area per bubble
///       texts: list[str],
///       page_width: int,
///       hints: list[TypesettingHint | None] | None = None,
///   ) -> RenderResult
///
/// Returns RGBA `[H, W, 4]` for direct codec encode without conversion.
#[pyfunction]
#[pyo3(signature = (clean, polygons, texts, page_width, hints=None))]
fn render<'py>(
    py: Python<'py>,
    clean:       PyReadonlyArray3<'py, u8>,
    polygons:    Vec<Vec<[f64; 2]>>,
    texts:       Vec<String>,
    page_width:  u32,
    hints:       Option<Vec<Option<TypesettingHint>>>,
) -> PyResult<RenderResult> {
    let (clean_rgba, cw, ch) = view_rgba(&clean)?;

    let hints_vec: Vec<Option<core::TypesettingHint>> = hints
        .unwrap_or_default()
        .into_iter()
        .map(|opt| opt.map(Into::into))
        .collect();

    let out = core::render_page_rgba(
        clean_rgba, cw, ch,
        polygons, texts, page_width, hints_vec,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let arr = ndarray::Array3::from_shape_vec(
        (out.height as usize, out.width as usize, 4),
        out.rgba,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(RenderResult {
        image:   arr.into_pyarray(py).unbind(),
        bubbles: out.bubbles.into_iter().map(Into::into).collect(),
    })
}

fn view_rgba(arr: &PyReadonlyArray3<'_, u8>) -> PyResult<(Vec<u8>, u32, u32)> {
    let a = arr.as_array();
    if a.shape().len() != 3 || a.shape()[2] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected RGBA [H, W, 4] uint8",
        ));
    }
    let h = a.shape()[0] as u32;
    let w = a.shape()[1] as u32;
    let data: Vec<u8> = a
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| a.iter().copied().collect());
    Ok((data, w, h))
}

/// Compute character budget for a bubble.
#[pyfunction]
#[pyo3(signature = (bubble_w, bubble_h, page_width, src_font_size_px=0, src_line_count=0))]
fn char_budget(
    bubble_w:         u32,
    bubble_h:         u32,
    page_width:       u32,
    src_font_size_px: u32,
    src_line_count:   u32,
) -> (u32, u32, u32) {
    core::char_budget(bubble_w, bubble_h, page_width, src_font_size_px, src_line_count)
}

#[pymodule]
fn typoon_render(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    m.add_function(wrap_pyfunction!(char_budget, m)?)?;
    m.add_function(wrap_pyfunction!(crate::stitch::stitch_pages, m)?)?;
    m.add_class::<RenderResult>()?;
    m.add_class::<BubbleInfo>()?;
    m.add_class::<TypesettingHint>()?;
    m.add_class::<crate::stitch::StitchResult>()?;
    Ok(())
}
