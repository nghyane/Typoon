use std::path::{Path, PathBuf};

use anyhow::Result;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tiff"];

/// Find all image files in a directory, sorted by name.
pub fn discover_images(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() && is_image_ext(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Load images from paths, failing on any error.
pub fn load_images(paths: &[PathBuf]) -> Result<Vec<image::DynamicImage>> {
    paths
        .iter()
        .map(|p| image::open(p).map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", p.display())))
        .collect()
}

/// Check if a directory contains at least one image file.
pub fn has_images(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| is_image_ext(&e.path()))
        })
        .unwrap_or(false)
}

fn is_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
}
