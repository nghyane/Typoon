//! Crate root for `typoon-render`.
//!
//! The render core lives in `core` and is feature-agnostic. Entry-point
//! modules wrap it in either a PyO3 extension (`python` feature, default)
//! or a wasm-bindgen cdylib (`wasm` feature, for the Cloudflare Workers
//! typeset stage).
//!
//! Only one of `python` / `wasm` can be enabled at a time; the build script
//! does not enforce this — Cargo's feature unification means downstream
//! must pick one explicitly (`--no-default-features --features wasm` for
//! the worker build).

pub mod core;
pub mod fit;
pub mod font;
pub mod layout;
pub mod overlay;
pub mod shape;
pub mod types;

// Legacy module — kept around because the Python entry's PyO3 `StitchResult`
// wraps a numpy ndarray and is structurally tied to the old API. The pure
// equivalent for non-Python callers is `core::stitch_pages`.
#[cfg(feature = "python")]
pub mod stitch;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "wasm")]
mod wasm;
