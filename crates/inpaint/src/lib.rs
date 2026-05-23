// SPDX-License-Identifier: GPL-3.0-or-later
//! typoon-inpaint — AOT-GAN inpaint service.
//!
//! Crate targets:
//!   - cdylib  → Python extension (PyO3, `python` feature)
//!   - rlib    → consumed by `bin/serve.rs` (CF container HTTP server)
//!
//! Module hierarchy:
//!   domain/    — pure data contracts (BBox, InpaintPlan, PadProfile, …)
//!   pipeline/  — orchestration: sources → close → regions → route → compose
//!   adapters/  — I/O-less compute: canny, rasterise, flat_fill, png_codec
//!   ffi/       — PyO3 wrappers (python feature only)
//!
//! The old `page.rs` / `s3.rs` split (mask-close + R2 I/O mixed) is gone.
//! `bin/serve.rs` now calls `pipeline::run_page` directly and owns all
//! async I/O itself.

pub mod adapters;
pub mod domain;
pub mod pipeline;

#[cfg(feature = "python")]
mod ffi;

// ── Re-export the Candle model so serve.rs and ffi can reach it ──────────

mod model;
pub use model::Inpainter;

// ── PyO3 module entry-point ───────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn typoon_inpaint(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register(m)
}
