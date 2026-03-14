use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;

/// Execution provider strategy for a lazy session.
#[derive(Debug, Clone, Default)]
pub enum SessionEp {
    /// CPU only — safe for all models, no hardware dependencies.
    #[default]
    Cpu,
    /// Try hardware acceleration at runtime with automatic fallback:
    /// CUDA (NVIDIA GPU) → CoreML (macOS ANE/GPU) → DirectML (Windows GPU) → CPU.
    Accelerated {
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

    /// Create a lazy session with hardware-accelerated EP.
    /// Tries CUDA → CoreML → DirectML → CPU at runtime.
    pub fn new_accelerated(model_path: PathBuf, cache_dir: Option<PathBuf>) -> Self {
        Self {
            model_path,
            ep: SessionEp::Accelerated { cache_dir },
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
                    SessionEp::Accelerated { cache_dir } => {
                        build_accelerated_session(&self.model_path, cache_dir.as_deref())
                    }
                };
                match result {
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

/// Try hardware EPs in priority order at runtime.
/// ort's `with_execution_providers` accepts multiple EPs — it tries each in order
/// and silently skips unavailable ones, falling back to CPU.
fn build_accelerated_session(
    model_path: &std::path::Path,
    cache_dir: Option<&std::path::Path>,
) -> ort::Result<Session> {
    use ort::ep;

    let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();

    // 1. CUDA — best for NVIDIA GPUs (RTX 4090, etc.)
    eps.push(ep::CUDA::default().build());

    // 2. CoreML — best for macOS Apple Silicon (ANE/GPU)
    {
        let mut coreml = ep::CoreML::default()
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_compute_units(ep::coreml::ComputeUnits::All)
            .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction);

        if let Some(dir) = cache_dir {
            std::fs::create_dir_all(dir).ok();
            coreml = coreml.with_model_cache_dir(dir.display().to_string());
        }

        eps.push(coreml.build());
    }

    // 3. DirectML — fallback for Windows GPUs (AMD, Intel, NVIDIA without CUDA)
    eps.push(ep::DirectML::default().build());

    // ort tries each EP in order, skips unavailable ones, falls back to CPU
    Session::builder()?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)
}
