use std::path::{Path, PathBuf};

use anyhow::Result;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tiff"];

pub fn discover_images(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("{} is not a directory", dir.display());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    paths.push(path);
                }
            }
        }
    }
    paths.sort();
    Ok(paths)
}

pub fn load_images(paths: &[PathBuf]) -> Result<Vec<image::DynamicImage>> {
    paths
        .iter()
        .map(|p| image::open(p).map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", p.display())))
        .collect()
}

pub fn has_images(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            })
        })
        .unwrap_or(false)
}

pub fn parse_chapter_number(name: &str) -> Option<usize> {
    name.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}
