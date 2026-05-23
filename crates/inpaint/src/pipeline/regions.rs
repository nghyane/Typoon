// SPDX-License-Identifier: GPL-3.0-or-later
//! Region merge: group scan BBoxes → inpaint regions (max 3 per page).

use crate::domain::{BBox, BlockClass, GroupMask};

const MERGE_DISTANCE: i32  = 96;
const MAX_REGIONS:    usize = 3;

#[derive(Debug)]
pub struct Region {
    pub bbox:           BBox,
    pub dominant_class: BlockClass,
}

/// Merge group bboxes by L∞ distance, cap at `MAX_REGIONS`.
pub fn merge(groups: &[GroupMask]) -> Vec<Region> {
    if groups.is_empty() { return Vec::new(); }

    // Seed: one region per group
    let mut regions: Vec<(BBox, BlockClass)> = groups.iter()
        .map(|g| (g.bbox, g.class))
        .collect();

    // Iterative merge until stable
    loop {
        let mut merged = false;
        'outer: for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                if regions[i].0.distance_to(&regions[j].0) <= MERGE_DISTANCE {
                    let bb = regions[i].0.union(&regions[j].0);
                    let cl = dominant(regions[i].1, regions[j].1);
                    regions[i] = (bb, cl);
                    regions.remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
        if !merged { break; }
    }

    // Cap: forcibly merge the two closest until ≤ MAX_REGIONS
    while regions.len() > MAX_REGIONS {
        let (mut bi, mut bj) = (0, 1);
        let mut best = i32::MAX;
        for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                let d = regions[i].0.distance_to(&regions[j].0);
                if d < best { best = d; bi = i; bj = j; }
            }
        }
        let merged_bb = regions[bi].0.union(&regions[bj].0);
        let merged_cl = dominant(regions[bi].1, regions[bj].1);
        regions[bi] = (merged_bb, merged_cl);
        regions.remove(bj);
    }

    regions.into_iter().map(|(bb, cl)| Region { bbox: bb, dominant_class: cl }).collect()
}

fn dominant(a: BlockClass, b: BlockClass) -> BlockClass {
    // Prefer dialogue > narration > sfx for mixed regions (conservative).
    use BlockClass::*;
    match (a, b) {
        (Dialogue, _) | (_, Dialogue) => Dialogue,
        (Narration, _) | (_, Narration) => Narration,
        _ => Sfx,
    }
}
