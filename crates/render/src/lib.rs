pub mod border;
pub mod fit;
pub mod layout;
pub mod overlay;
pub mod stitch;
pub mod types;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

/// Render translated text onto an erased page.
///
/// Input/output: numpy RGB uint8 [H, W, 3]. Zero PNG overhead.
#[pyfunction]
fn render<'py>(
    py: Python<'py>,
    original: PyReadonlyArray3<'py, u8>,
    clean: PyReadonlyArray3<'py, u8>,
    polygons: Vec<Vec<[f64; 2]>>,
    texts: Vec<String>,
    page_width: u32,
) -> PyResult<RenderResult> {
    if polygons.len() != texts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polygons and texts must have same length",
        ));
    }

    let original_img = rgb_array_to_image(&original)?;
    let clean_img = rgb_array_to_image(&clean)?;

    // Border detect + build drawable areas
    let bubbles: Vec<overlay::RenderBubble> = polygons
        .iter()
        .zip(&texts)
        .filter(|(_, text)| !text.is_empty())
        .map(|(poly, text)| {
            let insets = border::detect_edge_insets(&original_img, poly);
            let area = layout::DrawableArea::from_polygon_insets(poly, insets);
            overlay::RenderBubble {
                translated_text: text.clone(),
                area,
                font_size_px: 0,
                line_height: 0.0,
            }
        })
        .collect();

    // Fit text (binary search font sizes + page normalization)
    let fit_items: Vec<(&str, &layout::DrawableArea)> = bubbles
        .iter()
        .map(|b| (b.translated_text.as_str(), &b.area))
        .collect();

    let fits = fit::FitEngine::fit_page_areas(&fit_items, page_width)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let fitted: Vec<overlay::RenderBubble> = bubbles
        .into_iter()
        .zip(&fits)
        .map(|(mut b, f)| {
            b.translated_text = f.text.clone();
            b.font_size_px = f.font_size_px;
            b.line_height = f.line_height;
            b
        })
        .collect();

    let info: Vec<BubbleInfo> = fitted
        .iter()
        .zip(&fits)
        .map(|(b, f)| BubbleInfo {
            font_size_px: f.font_size_px,
            line_height: f.line_height,
            overflow: f.overflow,
            rect: {
                let (x, y, w, h) = b.area.rect();
                [x, y, w, h]
            },
        })
        .collect();

    // Render text → RGBA → extract RGB via single memcpy
    let rendered = overlay::render(&clean_img, &fitted);
    let (w, h) = rendered.dimensions();
    let rgba_buf = rendered.into_raw(); // Vec<u8> RGBA, contiguous

    // RGBA → RGB: one pass, pre-allocated
    let pixel_count = (h * w) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        rgb.push(rgba_buf[i * 4]);
        rgb.push(rgba_buf[i * 4 + 1]);
        rgb.push(rgba_buf[i * 4 + 2]);
    }

    let arr = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), rgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(RenderResult {
        image: arr.into_pyarray(py).unbind(),
        bubbles: info,
    })
}

/// Convert numpy RGB [H,W,3] → DynamicImage. Single memcpy via as_slice().
fn rgb_array_to_image(arr: &PyReadonlyArray3<'_, u8>) -> PyResult<image::DynamicImage> {
    let a = arr.as_array();
    let shape = a.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;

    // Contiguous C-order → single memcpy
    let data: Vec<u8> = if let Some(slice) = a.as_slice() {
        slice.to_vec()
    } else {
        // Non-contiguous fallback (rare — e.g. transposed arrays)
        a.iter().copied().collect()
    };

    image::RgbImage::from_raw(w, h, data)
        .map(image::DynamicImage::ImageRgb8)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid image dimensions"))
}

#[pyclass]
struct RenderResult {
    #[pyo3(get)]
    image: Py<PyArray3<u8>>,
    #[pyo3(get)]
    bubbles: Vec<BubbleInfo>,
}

#[pyclass]
#[derive(Clone)]
struct BubbleInfo {
    #[pyo3(get)]
    font_size_px: u32,
    #[pyo3(get)]
    line_height: f64,
    #[pyo3(get)]
    overflow: bool,
    #[pyo3(get)]
    rect: [f64; 4],
}

#[pymodule]
fn typoon_render(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    m.add_function(wrap_pyfunction!(stitch::stitch_pages, m)?)?;
    m.add_class::<RenderResult>()?;
    m.add_class::<BubbleInfo>()?;
    m.add_class::<stitch::StitchResult>()?;
    Ok(())
}
