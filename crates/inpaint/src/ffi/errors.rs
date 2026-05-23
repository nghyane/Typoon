// SPDX-License-Identifier: GPL-3.0-or-later
//! anyhow → Python exception mapping.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

pub fn to_pyerr(e: anyhow::Error) -> PyErr {
    // Build full chain
    let mut msg = format!("{e}");
    for cause in e.chain().skip(1) {
        msg.push_str(&format!("\n  caused by: {cause}"));
    }
    // Classify
    if let Some(io) = e.downcast_ref::<std::io::Error>() {
        if io.kind() == std::io::ErrorKind::NotFound {
            return PyFileNotFoundError::new_err(msg);
        }
    }
    if msg.contains("msgpack") || msg.contains("decode") || msg.contains("degenerate") {
        return PyValueError::new_err(msg);
    }
    PyRuntimeError::new_err(msg)
}
