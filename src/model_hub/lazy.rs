use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;

/// Lazy-loading ONNX session wrapper.
///
/// Stores the model path at construction time (zero cost). The actual ONNX
/// `Session` is created on the first `.get()` call and cached for subsequent
/// uses. This avoids creating CoreML contexts at startup for models that may
/// not be needed in a given run.
pub struct LazySession {
    model_path: PathBuf,
    session: OnceLock<Option<Mutex<Session>>>,
}

impl LazySession {
    /// Create a lazy session from a resolved model path.
    /// Does NOT load the ONNX model — that happens on first `.get()`.
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            session: OnceLock::new(),
        }
    }

    /// Get the session behind a Mutex, loading on first call.
    /// Returns `None` if initialization failed.
    pub fn get(&self) -> Option<&Mutex<Session>> {
        self.session
            .get_or_init(|| {
                match Session::builder().and_then(|mut b| b.commit_from_file(&self.model_path)) {
                    Ok(session) => {
                        tracing::info!(
                            "LazySession loaded: {}",
                            self.model_path.display()
                        );
                        Some(Mutex::new(session))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "LazySession failed to load {}: {e}",
                            self.model_path.display()
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Check if already loaded (without triggering load).
    pub fn is_loaded(&self) -> bool {
        self.session.get().is_some_and(|opt| opt.is_some())
    }

    /// Path to the ONNX model file.
    pub fn path(&self) -> &std::path::Path {
        &self.model_path
    }
}
