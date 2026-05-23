// SPDX-License-Identifier: GPL-3.0-or-later
//! JPEG / PNG codec helpers.

use anyhow::{Context, Result};
use image::{ImageBuffer, ImageEncoder, Rgb};
use image::codecs::png::PngEncoder;
use std::io::Cursor;

pub fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes)
        .context("JPEG decode failed")?
        .to_rgb8();
    let w = img.width() as usize;
    let h = img.height() as usize;
    Ok((img.into_raw(), w, h))
}

pub fn encode_png(rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let buf: ImageBuffer<Rgb<u8>, &[u8]> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
        .context("RGB buffer shape mismatch")?;
    let mut out = Vec::with_capacity(w * h * 3);
    PngEncoder::new(Cursor::new(&mut out))
        .write_image(buf.as_raw(), w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .context("PNG encode failed")?;
    Ok(out)
}
