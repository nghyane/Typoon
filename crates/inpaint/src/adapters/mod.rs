// SPDX-License-Identifier: GPL-3.0-or-later
//! Concrete mask-source adapters.
//! Each module implements `MaskSource` trait from `pipeline/sources.rs`.

pub mod canny;
pub mod flat_fill;
pub mod png_codec;
pub mod rasterise;
