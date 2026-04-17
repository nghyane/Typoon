//! Stitch pages into a single vertical strip.
//!
//! Wide outlier pages resized to median width. Single allocation.
//! Peak memory: one output buffer — input pages are read sequentially
//! and can be GC'd by Python between calls.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

#[pyclass]
pub struct StitchResult {
    #[pyo3(get)]
    pub image: Py<PyArray3<u8>>,
    #[pyo3(get)]
    pub heights: Vec<u32>,
    #[pyo3(get)]
    pub target_width: u32,
}

/// Stitch pages into one vertical strip with median-width normalization.
#[pyfunction]
pub fn stitch_pages<'py>(
    py: Python<'py>,
    pages: Vec<PyReadonlyArray3<'py, u8>>,
) -> PyResult<StitchResult> {
    if pages.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no pages"));
    }

    let dims: Vec<(usize, usize)> = pages
        .iter()
        .map(|p| {
            let s = p.as_array();
            (s.shape()[0], s.shape()[1])
        })
        .collect();

    // Single page — no stitch needed
    if pages.len() == 1 {
        let (h, w) = dims[0];
        let data: Vec<u8> = pages[0].as_array().iter().copied().collect();
        let arr = ndarray::Array3::from_shape_vec((h, w, 3), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return Ok(StitchResult {
            image: arr.into_pyarray(py).unbind(),
            heights: vec![h as u32],
            target_width: w as u32,
        });
    }

    // Median width
    let mut widths: Vec<usize> = dims.iter().map(|d| d.1).collect();
    widths.sort_unstable();
    let target_w = widths[widths.len() / 2];

    // Output heights (wide pages scaled down)
    let out_heights: Vec<usize> = dims
        .iter()
        .map(|&(h, w)| {
            if w > target_w {
                ((h as f64) * (target_w as f64) / (w as f64)).round() as usize
            } else {
                h
            }
        })
        .collect();

    let total_h: usize = out_heights.iter().sum();

    // Single allocation, white-filled
    let mut buf = vec![255u8; total_h * target_w * 3];

    let mut y_off: usize = 0;
    for (i, page) in pages.iter().enumerate() {
        let a = page.as_array();
        let (src_h, src_w) = dims[i];
        let dst_h = out_heights[i];

        if src_w == target_w {
            // Direct copy
            if let Some(src) = a.as_slice() {
                for row in 0..src_h.min(dst_h) {
                    let s = row * src_w * 3;
                    let d = (y_off + row) * target_w * 3;
                    let n = src_w * 3;
                    buf[d..d + n].copy_from_slice(&src[s..s + n]);
                }
            }
        } else if src_w > target_w {
            // Bilinear resize down
            let sx = src_w as f64 / target_w as f64;
            let sy = src_h as f64 / dst_h as f64;
            if let Some(src) = a.as_slice() {
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
            }
        } else {
            // Narrow: copy + white pad (already white from init)
            if let Some(src) = a.as_slice() {
                for row in 0..src_h {
                    let s = row * src_w * 3;
                    let d = (y_off + row) * target_w * 3;
                    let n = src_w * 3;
                    buf[d..d + n].copy_from_slice(&src[s..s + n]);
                }
            }
        }

        y_off += dst_h;
    }

    let arr = ndarray::Array3::from_shape_vec((total_h, target_w, 3), buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(StitchResult {
        image: arr.into_pyarray(py).unbind(),
        heights: out_heights.iter().map(|&h| h as u32).collect(),
        target_width: target_w as u32,
    })
}
