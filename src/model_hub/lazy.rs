use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::session::Session;

/// Lazy-loading ONNX session wrapper.
///
/// The actual `Session` is created on the first `.get()` call and cached.
/// `Session::run(&self)` is thread-safe (ONNX Runtime guarantees concurrent
/// inference on the same session), so no Mutex is needed.
pub struct LazySession {
    model_path: PathBuf,
    gpu: bool,
    session: OnceLock<Option<Session>>,
}

impl LazySession {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            gpu: false,
            session: OnceLock::new(),
        }
    }

    pub fn gpu(model_path: PathBuf) -> Self {
        Self {
            model_path,
            gpu: true,
            session: OnceLock::new(),
        }
    }

    /// Get the session, loading on first call.
    pub fn get(&self) -> Option<&Session> {
        self.session.get_or_init(|| self.load_session()).as_ref()
    }

    fn load_session(&self) -> Option<Session> {
        if self.gpu {
            load_prefer_gpu(&self.model_path)
        } else {
            load_cpu(&self.model_path, false)
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.get().is_some_and(|opt| opt.is_some())
    }

    pub fn path(&self) -> &std::path::Path {
        &self.model_path
    }
}

fn load_prefer_gpu(model_path: &Path) -> Option<Session> {
    match build_gpu_session(model_path) {
        Ok(session) => {
            tracing::info!("LazySession loaded (GPU): {}", model_path.display());
            Some(session)
        }
        Err(gpu_err) => {
            tracing::warn!(
                "LazySession GPU init failed for {}: {gpu_err}; falling back to CPU",
                model_path.display()
            );
            load_cpu(model_path, true)
        }
    }
}

fn load_cpu(model_path: &Path, is_fallback: bool) -> Option<Session> {
    match build_cpu_session(model_path) {
        Ok(session) => {
            if is_fallback {
                tracing::info!(
                    "LazySession loaded (CPU fallback): {}",
                    model_path.display()
                );
            } else {
                tracing::info!("LazySession loaded: {}", model_path.display());
            }
            Some(session)
        }
        Err(cpu_err) => {
            if is_fallback {
                tracing::warn!(
                    "LazySession CPU fallback also failed for {}: {cpu_err}",
                    model_path.display()
                );
            } else {
                tracing::warn!(
                    "LazySession failed to load {}: {cpu_err}",
                    model_path.display()
                );
            }
            None
        }
    }
}

fn build_gpu_session(model_path: &Path) -> ort::Result<Session> {
    #[cfg(target_os = "macos")]
    {
        let cache_dir = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("ep_cache");
        std::fs::create_dir_all(&cache_dir).ok();

        match build_coreml_session(
            model_path,
            &cache_dir,
            ort::ep::coreml::ModelFormat::MLProgram,
        ) {
            Ok(session) => Ok(session),
            Err(ml_program_err) => {
                tracing::warn!(
                    "CoreML MLProgram init failed for {}: {ml_program_err}; retrying NeuralNetwork format",
                    model_path.display()
                );
                build_coreml_session(
                    model_path,
                    &cache_dir,
                    ort::ep::coreml::ModelFormat::NeuralNetwork,
                )
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use ort::ep;
        use ort::session::builder::AutoDevicePolicy;

        let mut eps = Vec::new();

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        eps.push(ep::CUDA::default().build());

        #[cfg(target_os = "windows")]
        eps.push(ep::DirectML::default().build());

        Session::builder()?
            .with_intra_op_spinning(true)?
            .with_inter_op_spinning(true)?
            .with_auto_device(AutoDevicePolicy::MaxPerformance)?
            .with_execution_providers(eps)?
            .commit_from_file(model_path)
    }
}

#[cfg(target_os = "macos")]
fn build_coreml_session(
    model_path: &Path,
    cache_dir: &Path,
    model_format: ort::ep::coreml::ModelFormat,
) -> ort::Result<Session> {
    use ort::ep;
    use ort::session::builder::AutoDevicePolicy;

    Session::builder()?
        .with_intra_op_spinning(true)?
        .with_inter_op_spinning(true)?
        .with_auto_device(AutoDevicePolicy::MaxPerformance)?
        .with_execution_providers(vec![
            ep::CoreML::default()
                .with_model_format(model_format)
                .with_compute_units(ep::coreml::ComputeUnits::All)
                .with_model_cache_dir(cache_dir.display().to_string())
                .build(),
        ])?
        .commit_from_file(model_path)
}

fn build_cpu_session(model_path: &Path) -> ort::Result<Session> {
    Session::builder()?.commit_from_file(model_path)
}
