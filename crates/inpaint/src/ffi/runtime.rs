// SPDX-License-Identifier: GPL-3.0-or-later
//! PyO3 classes and functions registered in the Python module.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;
use std::sync::Arc;

use crate::Inpainter;
use crate::pipeline;
use super::errors::to_pyerr;

/// Stateful Python class. Holds the warm Candle session so Python CLI
/// can process N pages without reloading the model.
///
/// ```python
/// from typoon_inpaint import InpaintRuntime
///
/// rt = InpaintRuntime(model_path="/app/model.safetensors")
/// png_bytes = rt.inpaint_page(jpeg_bytes, plan_bytes)
/// # or with debug artifacts:
/// png_bytes = rt.inpaint_page(jpeg_bytes, plan_bytes, debug_dir="/tmp/debug")
/// ```
#[pyclass(name = "InpaintRuntime")]
pub struct PyInpaintRuntime {
    inpainter: Arc<Inpainter>,
}

#[pymethods]
impl PyInpaintRuntime {
    #[new]
    #[pyo3(signature = (model_path, fp16 = false))]
    fn new(model_path: &str, fp16: bool) -> PyResult<Self> {
        let inpainter = Inpainter::load(std::path::Path::new(model_path), fp16)
            .map_err(to_pyerr)?;
        Ok(Self { inpainter: Arc::new(inpainter) })
    }

    /// Inpaint one page. Releases the GIL during all CPU work.
    ///
    /// Args:
    ///     jpeg_bytes: raw JPEG of the prepared page
    ///     scan_bytes: msgpack bytes — either a bare InpaintPlan or the full
    ///                 scan msgpack with embedded `inpaint_plan` field
    ///     debug_dir:  optional directory; intermediate PNGs written there
    ///
    /// Returns:
    ///     PNG bytes of the inpainted page
    #[pyo3(signature = (jpeg_bytes, scan_bytes, debug_dir = None))]
    fn inpaint_page<'py>(
        &self,
        py:        Python<'py>,
        jpeg_bytes: &[u8],
        scan_bytes: &[u8],
        debug_dir:  Option<&str>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let jpeg  = jpeg_bytes.to_vec();
        let scan  = scan_bytes.to_vec();
        let dbg   = debug_dir.map(PathBuf::from);
        let inp   = self.inpainter.clone();

        let png = py.allow_threads(move || {
            pipeline::run_page(&inp, jpeg, scan, dbg.as_deref())
        }).map_err(to_pyerr)?;

        Ok(PyBytes::new_bound(py, &png))
    }
}

// ── Stateless helpers (for probe scripts) ────────────────────────────────

/// Decode an InpaintPlan msgpack → Python dict. No model needed.
///
/// Returns a dict matching the JSON schema (page_kind, page_size, groups, …).
#[pyfunction]
fn decode_plan(py: Python<'_>, plan_bytes: &[u8]) -> PyResult<PyObject> {
    use crate::domain::InpaintPlan;
    let plan = InpaintPlan::from_msgpack(plan_bytes).map_err(to_pyerr)?;

    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("page_index", plan.page_index)?;
    d.set_item("page_size",  vec![plan.page_size[0], plan.page_size[1]])?;
    d.set_item("page_kind",  format!("{:?}", plan.page_kind).to_lowercase())?;
    let groups: Vec<_> = plan.groups.iter().map(|g| {
        let gd = pyo3::types::PyDict::new_bound(py);
        let _ = gd.set_item("idx",        g.idx);
        let _ = gd.set_item("bbox",       vec![g.bbox.x1, g.bbox.y1, g.bbox.x2, g.bbox.y2]);
        let _ = gd.set_item("origin",     format!("{:?}", g.origin).to_lowercase().replace("lens", "lens_"));
        let _ = gd.set_item("class",      format!("{:?}", g.class).to_lowercase());
        let _ = gd.set_item("shape_kind", format!("{:?}", g.shape_kind).to_lowercase());
        let _ = gd.set_item("n_polygons", g.polygons.len());
        let _ = gd.set_item("n_rasters",  g.rasters.len());
        gd
    }).collect();
    d.set_item("groups", groups)?;
    Ok(d.into())
}

/// Build page mask from plan + JPEG. Returns raw bytes (W*H, 0 or 255).
/// Useful for probe scripts to inspect mask without running AOT.
#[pyfunction]
#[pyo3(signature = (plan_bytes, jpeg_bytes, debug_dir = None))]
fn rasterise_plan_mask<'py>(
    py:        Python<'py>,
    plan_bytes: &[u8],
    jpeg_bytes: &[u8],
    debug_dir:  Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let plan_b = plan_bytes.to_vec();
    let jpeg_b = jpeg_bytes.to_vec();
    let dbg    = debug_dir.map(PathBuf::from);

    let mask_bytes = py.allow_threads(move || -> anyhow::Result<Vec<u8>> {
        use crate::domain::InpaintPlan;
        use crate::adapters::png_codec;
        use crate::pipeline::{sources, DebugSink};

        let plan = InpaintPlan::from_msgpack(&plan_b)?;
        let (rgb, w, h) = png_codec::decode_jpeg(&jpeg_b)?;
        let dbg  = DebugSink::from_path(dbg.as_deref());
        let mask = sources::build_page_mask(&rgb, w, h, &plan, &dbg)?;
        Ok(mask)
    }).map_err(to_pyerr)?;

    Ok(PyBytes::new_bound(py, &mask_bytes))
}

// ── Module registration ───────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInpaintRuntime>()?;
    m.add_function(wrap_pyfunction!(decode_plan, m)?)?;
    m.add_function(wrap_pyfunction!(rasterise_plan_mask, m)?)?;
    Ok(())
}
