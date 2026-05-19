// SPDX-License-Identifier: GPL-3.0-or-later
//! PyO3 extension entry-point.
//!
//! Mirrors the pattern from `crates/render/src/py.rs`:
//!   numpy arrays in → model forward → numpy array out
//!
//! Python usage:
//!   import typoon_inpaint
//!   out = typoon_inpaint.inpaint(image_rgb, mask, model_path)
//!
//! where:
//!   image_rgb: np.ndarray[uint8] shape (H, W, 3)  RGB
//!   mask:      np.ndarray[uint8] shape (H, W)      >=127 = inpaint
//!   model_path: str  path to model.safetensors
//!
//! Returns np.ndarray[uint8] shape (H, W, 3) RGB.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::Inpainter;

// ─── Lazy model cache — one per process, loaded on first call ──────────────

struct ModelCache {
    inpainter: Inpainter,
    path:      String,
}

static CACHE: OnceLock<Mutex<Option<ModelCache>>> = OnceLock::new();

fn get_or_load(model_path: &str) -> PyResult<std::sync::MutexGuard<'static, Option<ModelCache>>> {
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("mutex poisoned: {e}"))
    })?;
    if guard.as_ref().map(|c| c.path.as_str()) != Some(model_path) {
        let inpainter = Inpainter::load(Path::new(model_path), false)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        *guard = Some(ModelCache { inpainter, path: model_path.to_string() });
    }
    Ok(guard)
}

// ──────────────────────────────────────────────────────────────────────────────

/// Inpaint masked region of an RGB image using AOT-GAN.
///
/// Args:
///     image_rgb:  uint8 ndarray (H, W, 3)
///     mask:       uint8 ndarray (H, W)   — >=127 marks pixels to fill
///     model_path: path to model.safetensors
///
/// Returns:
///     uint8 ndarray (H, W, 3) — same shape as input, masked region filled
#[pyfunction]
fn inpaint<'py>(
    py:         Python<'py>,
    image_rgb:  PyReadonlyArray3<'py, u8>,
    mask:       PyReadonlyArray2<'py, u8>,
    model_path: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_arr  = image_rgb.as_array();
    let mask_arr = mask.as_array();

    let shape = img_arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image_rgb must be uint8 (H, W, 3)",
        ));
    }
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    if mask_arr.shape() != [h as usize, w as usize] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mask shape must match image (H, W)",
        ));
    }

    let image_bytes: Vec<u8> = img_arr
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| img_arr.iter().copied().collect());
    let mask_bytes: Vec<u8> = mask_arr
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| mask_arr.iter().copied().collect());

    let guard = get_or_load(model_path)?;
    let inpainter = &guard.as_ref().unwrap().inpainter;

    let result = inpainter.inpaint(&image_bytes, &mask_bytes, w, h)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let arr = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray(py))
}

#[pymodule]
fn typoon_inpaint(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inpaint, m)?)?;
    Ok(())
}
