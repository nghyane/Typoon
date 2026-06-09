// SPDX-License-Identifier: GPL-3.0-or-later
//! Flat-fill probe: sample bbox edges/corners and require background consensus.

use crate::domain::BBox;

const RING_PX:           usize = 14;
const MIN_ZONE_SAMPLES:  usize = 24;
const MAX_ZONE_SPREAD:   f32   = 34.0;
const MAX_INK_RATIO:     f32   = 0.18;
const CONSENSUS_DIST:    f32   = 24.0;
const MIN_EDGE_VOTES:    usize = 3;
const MIN_ALL_VOTES:     usize = 4;

/// Sample unmasked pixels around bbox edges/corners.
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
    if x1 <= x0 || y1 <= y0 { return None; }

    let edge_boxes = edge_zones(x0, y0, x1, y1, w, h);
    let mut zones = collect_zone_stats(rgb, mask, w, &edge_boxes);
    if let Some(color) = consensus(&zones, MIN_EDGE_VOTES) {
        return Some(color);
    }

    let corner_boxes = corner_zones(x0, y0, x1, y1, w, h);
    zones.extend(collect_zone_stats(rgb, mask, w, &corner_boxes));
    consensus(&zones, MIN_ALL_VOTES)
}

#[derive(Clone)]
struct ZoneStats {
    median: [u8; 3],
    samples: Vec<[u8; 3]>,
}

fn edge_zones(x0: usize, y0: usize, x1: usize, y1: usize, w: usize, h: usize) -> Vec<(usize, usize, usize, usize)> {
    let r = RING_PX.min((x1 - x0 + 1) / 3).min((y1 - y0 + 1) / 3).max(1);
    let mut out = Vec::with_capacity(4);
    let _ = (w, h);
    out.push((x0, y0, x1, (y0 + r - 1).min(y1)));
    out.push((x0, y1.saturating_sub(r - 1), x1, y1));
    out.push((x0, y0, (x0 + r - 1).min(x1), y1));
    out.push((x1.saturating_sub(r - 1), y0, x1, y1));
    out
}

fn corner_zones(x0: usize, y0: usize, x1: usize, y1: usize, w: usize, h: usize) -> Vec<(usize, usize, usize, usize)> {
    let r = RING_PX.min((x1 - x0 + 1) / 3).min((y1 - y0 + 1) / 3).max(1);
    let _ = (w, h);
    vec![
        (x0, y0, (x0 + r - 1).min(x1), (y0 + r - 1).min(y1)),
        (x1.saturating_sub(r - 1), y0, x1, (y0 + r - 1).min(y1)),
        (x0, y1.saturating_sub(r - 1), (x0 + r - 1).min(x1), y1),
        (x1.saturating_sub(r - 1), y1.saturating_sub(r - 1), x1, y1),
    ]
}

fn collect_zone_stats(
    rgb: &[u8], mask: &[u8], w: usize,
    boxes: &[(usize, usize, usize, usize)],
) -> Vec<ZoneStats> {
    boxes.iter().filter_map(|&(x0, y0, x1, y1)| {
        if x1 < x0 || y1 < y0 { return None; }
        let mut samples = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * w + x;
                if mask[i] >= 127 { continue; }
                let p = i * 3;
                samples.push([rgb[p], rgb[p + 1], rgb[p + 2]]);
            }
        }
        zone_stats(samples)
    }).collect()
}

fn zone_stats(mut samples: Vec<[u8; 3]>) -> Option<ZoneStats> {
    if samples.len() < MIN_ZONE_SAMPLES { return None; }
    let median = median_rgb(&samples);
    let spread = percentile_spread(&mut samples);
    if spread > MAX_ZONE_SPREAD { return None; }
    if ink_ratio(&samples, median) > MAX_INK_RATIO { return None; }
    Some(ZoneStats { median, samples })
}

fn consensus(zones: &[ZoneStats], min_votes: usize) -> Option<[u8; 3]> {
    if zones.len() < min_votes { return None; }
    let mut best: Vec<&ZoneStats> = Vec::new();
    for z in zones {
        let cluster: Vec<&ZoneStats> = zones.iter()
            .filter(|o| color_dist(z.median, o.median) <= CONSENSUS_DIST)
            .collect();
        if cluster.len() > best.len() { best = cluster; }
    }
    if best.len() < min_votes { return None; }
    let mut samples = Vec::new();
    for z in best { samples.extend_from_slice(&z.samples); }
    Some(median_rgb(&samples))
}

fn median_rgb(samples: &[[u8; 3]]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let mut vals: Vec<u8> = samples.iter().map(|s| s[c]).collect();
        vals.sort_unstable();
        out[c] = vals[vals.len() / 2];
    }
    out
}

fn percentile_spread(samples: &mut [[u8; 3]]) -> f32 {
    let mut total = 0.0;
    for c in 0..3 {
        let mut vals: Vec<u8> = samples.iter().map(|s| s[c]).collect();
        vals.sort_unstable();
        let p10 = vals[vals.len() / 10] as f32;
        let p90 = vals[(vals.len() * 9 / 10).min(vals.len() - 1)] as f32;
        total += p90 - p10;
    }
    total / 3.0
}

fn ink_ratio(samples: &[[u8; 3]], bg: [u8; 3]) -> f32 {
    let bg_l = luminance(bg);
    let ink = samples.iter().filter(|&&p| {
        let l = luminance(p);
        let sat = p.iter().max().unwrap() - p.iter().min().unwrap();
        l + 38.0 < bg_l || sat > 72 || color_dist(p, bg) > 58.0
    }).count();
    ink as f32 / samples.len() as f32
}

fn color_dist(a: [u8; 3], b: [u8; 3]) -> f32 {
    let dr = a[0] as f32 - b[0] as f32;
    let dg = a[1] as f32 - b[1] as f32;
    let db = a[2] as f32 - b[2] as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn luminance(p: [u8; 3]) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}
