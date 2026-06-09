// SPDX-License-Identifier: GPL-3.0-or-later
//! Region merge: group scan BBoxes → inpaint regions.

use crate::domain::{BBox, BlockClass, Group};

const MERGE_DISTANCE: i32  = 96;
const MAX_MERGED_EDGE: i32 = 720;
const MAX_MERGED_AREA: i64 = 360_000;

#[derive(Debug)]
pub struct Region {
    pub bbox:           BBox,
    pub dominant_class: BlockClass,
}

/// Merge nearby group bboxes, but never force far/large regions into one tile.
pub fn merge(groups: &[Group]) -> Vec<Region> {
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
                let bb = regions[i].0.union(&regions[j].0);
                if regions[i].0.distance_to(&regions[j].0) <= MERGE_DISTANCE
                    && merge_is_safe(&bb)
                {
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

    regions.into_iter().map(|(bb, cl)| Region { bbox: bb, dominant_class: cl }).collect()
}

fn merge_is_safe(bb: &BBox) -> bool {
    bb.width().max(bb.height()) <= MAX_MERGED_EDGE && bb.area() <= MAX_MERGED_AREA
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
