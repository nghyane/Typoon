// SPDX-License-Identifier: GPL-3.0-or-later
//! Flat-fill probe: sample border pixels adjacent to the mask and decide
//! whether the background is uniform enough to skip AOT.

use crate::domain::BBox;

const FLAT_STD_THRESHOLD: f32 = 10.0;
const FLAT_MIN_SAMPLES:   usize = 32;

/// Sample unmasked pixels adjacent to the mask inside `bb`.
/// Returns `Some([r,g,b])` when the background is flat enough.
pub fn probe(
    rgb:  &[u8],
    mask: &[u8],
    w:    usize,
    h:    usize,
    bb:   &BBox,
) -> Option<[u8; 3]> {
    let x0 = bb.x1.max(0) as usize;
    let y0 = bb.y1.max(0) as usize;
    let x1 = bb.x2.min(w as i32 - 1) as usize;
    let y1 = bb.y2.min(h as i32 - 1) as usize;

    let mut samples: Vec<[u8; 3]> = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if mask[i] >= 127 { continue; }
            let adjacent =
                (x > 0     && mask[i - 1] >= 127) ||
                (x + 1 < w && mask[i + 1] >= 127) ||
                (y > 0     && mask[i - w] >= 127) ||
                (y + 1 < h && mask[i + w] >= 127);
            if !adjacent { continue; }
            let p = i * 3;
            samples.push([rgb[p], rgb[p+1], rgb[p+2]]);
        }
    }
    if samples.len() < FLAT_MIN_SAMPLES { return None; }

    let n = samples.len() as f32;
    let mut mean = [0f32; 3];
    for s in &samples {
        mean[0] += s[0] as f32; mean[1] += s[1] as f32; mean[2] += s[2] as f32;
    }
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    let var: f32 = samples.iter().map(|s| {
        (0..3).map(|c| { let d = s[c] as f32 - mean[c]; d*d }).sum::<f32>()
    }).sum::<f32>() / (n * 3.0);

    if var.sqrt() > FLAT_STD_THRESHOLD { return None; }
    Some([mean[0].round() as u8, mean[1].round() as u8, mean[2].round() as u8])
}
