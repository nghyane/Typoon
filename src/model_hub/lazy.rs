use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;

/// Execution provider strategy for a lazy session.
#[derive(Debug, Clone, Default)]
pub enum SessionEp {
    /// CPU only — no GPU/ANE overhead, best for models with many
    /// unsupported CoreML ops (high partition count).
    #[default]
    Cpu,
    /// CoreML with MLProgram format — uses ANE/GPU on Apple Silicon.
    /// Compiled models are cached to `cache_dir` to avoid recompilation.
    /// Best for models with good CoreML op coverage.
    CoreMl {
        cache_dir: Option<PathBuf>,
    },
}

/// Lazy-loading ONNX session wrapper.
///
/// Stores the model path and EP config at construction time (zero cost).
/// The actual ONNX `Session` is created on the first `.get()` call and
/// cached for subsequent uses.
pub struct LazySession {
    model_path: PathBuf,
    ep: SessionEp,
    session: OnceLock<Option<Mutex<Session>>>,
}

impl LazySession {
    /// Create a lazy session with CPU EP (default, safe for all models).
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            ep: SessionEp::Cpu,
            session: OnceLock::new(),
        }
    }

    /// Create a lazy session with CoreML EP (MLProgram format).
    /// Uses ANE/GPU on Apple Silicon for models with good op coverage.
    pub fn new_coreml(model_path: PathBuf, cache_dir: Option<PathBuf>) -> Self {
        Self {
            model_path,
            ep: SessionEp::CoreMl { cache_dir },
            session: OnceLock::new(),
        }
    }

    /// Get the session behind a Mutex, loading on first call.
    /// Returns `None` if initialization failed.
    pub fn get(&self) -> Option<&Mutex<Session>> {
        self.session
            .get_or_init(|| {
                let result = match &self.ep {
                    SessionEp::Cpu => {
                        Session::builder().and_then(|mut b| b.commit_from_file(&self.model_path))
                    }
                    SessionEp::CoreMl { cache_dir } => {
                        build_coreml_session(&self.model_path, cache_dir.as_deref())
                    }
                };
                match result {
                    Ok(session) => {
                        tracing::info!(
                            "LazySession loaded ({}): {}",
                            ep_label(&self.ep),
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

/// Build a session with CoreML EP (MLProgram format + optional cache).
fn build_coreml_session(
    model_path: &std::path::Path,
    cache_dir: Option<&std::path::Path>,
) -> ort::Result<Session> {
    use ort::ep;

    let mut coreml = ep::CoreML::default()
        .with_model_format(ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ep::coreml::ComputeUnits::All)
        .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction);

    if let Some(dir) = cache_dir {
        std::fs::create_dir_all(dir).ok();
        coreml = coreml.with_model_cache_dir(dir.display().to_string());
    }

    Session::builder()?
        .with_execution_providers([coreml.build()])?
        .commit_from_file(model_path)
}

fn ep_label(ep: &SessionEp) -> &'static str {
    match ep {
        SessionEp::Cpu => "cpu",
        SessionEp::CoreMl { .. } => "coreml",
    }
}
