use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;

/// Lazy-loading ONNX session wrapper.
///
/// Stores the model path at construction time (zero cost).
/// The actual ONNX `Session` is created on the first `.get()` call and
/// cached for subsequent uses.
///
/// Use `gpu()` to request hardware acceleration (CUDA/CoreML/DirectML).
pub struct LazySession {
    model_path: PathBuf,
    gpu: bool,
    session: OnceLock<Option<Mutex<Session>>>,
}

impl LazySession {
    /// Create a session using CPU only.
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            gpu: false,
            session: OnceLock::new(),
        }
    }

    /// Create a session with GPU acceleration (CoreML/CUDA/DirectML).
    pub fn gpu(model_path: PathBuf) -> Self {
        Self {
            model_path,
            gpu: true,
            session: OnceLock::new(),
        }
    }

    /// Get the session behind a Mutex, loading on first call.
    /// Returns `None` if initialization failed.
    pub fn get(&self) -> Option<&Mutex<Session>> {
        self.session
            .get_or_init(|| {
                let use_gpu = self.gpu;

                let result = if use_gpu {
                    build_gpu_session(&self.model_path)
                } else {
                    Session::builder()
                        .and_then(|mut b| b.commit_from_file(&self.model_path))
                };

                let label = if use_gpu { " (GPU)" } else { "" };
                match result {
                    Ok(session) => {
                        tracing::info!(
                            "LazySession loaded{label}: {}",
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

/// Build a session with GPU execution providers.
fn build_gpu_session(model_path: &std::path::Path) -> ort::Result<Session> {
    use ort::ep;

    let cache_dir = model_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("ep_cache");
    std::fs::create_dir_all(&cache_dir).ok();

    let mut eps = Vec::new();

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    eps.push(ep::CUDA::default().build());

    #[cfg(target_os = "macos")]
    eps.push(
        ep::CoreML::default()
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_compute_units(ep::coreml::ComputeUnits::All)
            .with_model_cache_dir(cache_dir.display().to_string())
            .build(),
    );

    #[cfg(target_os = "windows")]
    eps.push(ep::DirectML::default().build());

    Session::builder()?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)
}
