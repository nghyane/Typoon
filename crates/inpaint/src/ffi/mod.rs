// SPDX-License-Identifier: GPL-3.0-or-later
//! PyO3 FFI layer — thin wrappers around `pipeline::run_page`.
//! No business logic here.

mod errors;
mod runtime;

pub use runtime::register;
