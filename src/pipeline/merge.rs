use crate::detection::{LocalTextMask, TextRegion};

/// A bubble with grouped text lines, ready for OCR + concat.
pub struct MergedBubble {
    /// Bounding polygon over all lines in this group
    pub polygon: Vec<[f64; 2]>,
    /// Text line crops sorted top-to-bottom
    pub lines: Vec<TextRegion>,
    /// Max detection confidence across lines
    pub confidence: f64,
    /// Composited text mask from all lines (union of per-line masks)
    pub mask: Option<LocalTextMask>,
}

/// Maximum absolute vertical gap (px) between two lines to allow merging.
/// Prevents merging lines from separate bubbles that happen to be aligned.
const MAX_VERTICAL_GAP_PX: f64 = 40.0;

/// Maximum absolute horizontal gap (px) between two lines to allow merging.
/// Prevents merging lines from side-by-side bubbles that share similar Y ranges.
const MAX_HORIZONTAL_GAP_PX: f64 = 30.0;

/// Group PP-OCR text lines into bubble groups by spatial proximity.
///
/// Two lines are grouped if:
/// - Horizontal overlap ≥ 40% (same column of text)
/// - Vertical gap < 1.5× max line height AND < MAX_VERTICAL_GAP_PX
///
/// Single tiny regions (< 30×20 px) are filtered as noise.
/// Result sorted top-to-bottom for reading order.
pub fn group_lines(lines: Vec<TextRegion>) -> Vec<MergedBubble> {
    if lines.is_empty() {
        return Vec::new();
    }

    let mut lines = lines;
    lines.sort_by(|a, b| top_y(&a.polygon).partial_cmp(&top_y(&b.polygon)).unwrap());

    // Union-Find grouping
    let n = lines.len();
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if find(&parent, i) == find(&parent, j) {
                continue;
            }

            let (ix1, iy1, ix2, iy2) = bbox(&lines[i].polygon);
            let (jx1, jy1, jx2, jy2) = bbox(&lines[j].polygon);
            let max_h = (iy2 - iy1).max(jy2 - jy1);

            let h_gap = if ix2 < jx1 { jx1 - ix2 } else if jx2 < ix1 { ix1 - jx2 } else { 0.0 };
            let v_gap = if iy2 < jy1 { jy1 - iy2 } else if jy2 < iy1 { iy1 - jy2 } else { 0.0 };

            // Horizontal gap: reject side-by-side lines from different bubbles
            if h_gap > MAX_HORIZONTAL_GAP_PX {
                continue;
            }

            // Vertical gap: must be within both relative and absolute limits
            if v_gap > max_h * 1.5 || v_gap > MAX_VERTICAL_GAP_PX {
                continue;
            }

            // Horizontal overlap: lines must share ≥40% X range to be in the same column
            if horizontal_overlap_ratio(&lines[i].polygon, &lines[j].polygon) < 0.4 {
                continue;
            }

            union(&mut parent, i, j);
        }
    }

    // Collect groups
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(find(&parent, i)).or_default().push(i);
    }

    let mut result = Vec::new();
    for (_, indices) in groups {
        let mut group_lines: Vec<TextRegion> = indices.into_iter()
            .map(|i| std::mem::replace(&mut lines[i], placeholder_region()))
            .collect();

        // Sort by reading order: group lines into rows (similar Y), sort left→right within rows
        sort_reading_order(&mut group_lines);

        let polygon = bounding_polygon(&group_lines);

        // Filter noise: too small, or every line is portrait-oriented.
        // Normal dialogue lines are always landscape (w > h).
        // SFX/sound effects produce portrait lines (h >= w) — if ALL lines
        // in a group are portrait, the whole group is SFX.
        let (x1, y1, x2, y2) = bbox(&polygon);
        if (x2 - x1) < 80.0 || (y2 - y1) < 35.0 {
            continue;
        }
        // NOTE: SFX filtering moved to pipeline/mod.rs where OCR confidence
        // is available for more accurate classification.
        let confidence = group_lines.iter()
            .map(|l| l.confidence)
            .fold(0.0_f64, f64::max);
        let mask = composite_masks(&group_lines);

        result.push(MergedBubble {
            polygon,
            lines: group_lines,
            confidence,
            mask,
        });
    }

    result.sort_by(|a, b| top_y(&a.polygon).partial_cmp(&top_y(&b.polygon)).unwrap());
    result
}

// ── Reading order sort ──

/// Sort lines into reading order: top→bottom by row, left→right within each row.
/// Lines whose vertical centers are within half the max line height are considered same row.
fn sort_reading_order(lines: &mut [TextRegion]) {
    if lines.len() <= 1 {
        return;
    }

    // Compute center-Y and height for each line
    let metas: Vec<(f64, f64, f64)> = lines.iter().map(|l| {
        let (x1, y1, _, y2) = bbox(&l.polygon);
        let cy = (y1 + y2) / 2.0;
        let h = y2 - y1;
        (cy, h, x1)
    }).collect();

    let max_h = metas.iter().map(|m| m.1).fold(0.0_f64, f64::max);
    let row_tolerance = max_h * 0.5;

    // Sort by center-Y first, then assign rows
    let mut indices: Vec<usize> = (0..lines.len()).collect();
    indices.sort_by(|&a, &b| metas[a].0.partial_cmp(&metas[b].0).unwrap());

    // Group into rows: consecutive lines with center-Y within tolerance
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut current_row = vec![indices[0]];
    let mut row_cy = metas[indices[0]].0;

    for &idx in &indices[1..] {
        if (metas[idx].0 - row_cy).abs() <= row_tolerance {
            current_row.push(idx);
        } else {
            rows.push(current_row);
            current_row = vec![idx];
            row_cy = metas[idx].0;
        }
    }
    rows.push(current_row);

    // Sort each row left→right by x1
    for row in &mut rows {
        row.sort_by(|&a, &b| metas[a].2.partial_cmp(&metas[b].2).unwrap());
    }

    // Rebuild sorted order
    let sorted_indices: Vec<usize> = rows.into_iter().flatten().collect();
    let mut sorted: Vec<TextRegion> = sorted_indices.into_iter()
        .map(|i| std::mem::replace(&mut lines[i], placeholder_region()))
        .collect();
    lines.swap_with_slice(&mut sorted);
}

// ── Geometry helpers ──

fn top_y(polygon: &[[f64; 2]]) -> f64 {
    polygon.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min)
}

/// Axis-aligned bounding box of a polygon. Public for SFX filtering in pipeline.
pub fn line_bbox(polygon: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    bbox(polygon)
}

fn bbox(polygon: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let (mut x1, mut y1) = (f64::INFINITY, f64::INFINITY);
    let (mut x2, mut y2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon {
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    (x1, y1, x2, y2)
}

fn horizontal_overlap_ratio(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let (ax1, _, ax2, _) = bbox(a);
    let (bx1, _, bx2, _) = bbox(b);
    let overlap = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let min_w = (ax2 - ax1).min(bx2 - bx1);
    if min_w > 0.0 { overlap / min_w } else { 0.0 }
}

fn bounding_polygon(regions: &[TextRegion]) -> Vec<[f64; 2]> {
    let (mut x1, mut y1) = (f64::INFINITY, f64::INFINITY);
    let (mut x2, mut y2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for r in regions {
        for p in &r.polygon {
            x1 = x1.min(p[0]);
            y1 = y1.min(p[1]);
            x2 = x2.max(p[0]);
            y2 = y2.max(p[1]);
        }
    }
    vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
}

fn find(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

/// Composite per-line text masks into a single bubble mask (logical OR).
fn composite_masks(lines: &[TextRegion]) -> Option<LocalTextMask> {
    let masks: Vec<&LocalTextMask> = lines.iter().filter_map(|l| l.mask.as_ref()).collect();
    if masks.is_empty() {
        return None;
    }
    if masks.len() == 1 {
        return Some(masks[0].clone());
    }

    // Union bbox of all masks
    let mut ux1 = u32::MAX;
    let mut uy1 = u32::MAX;
    let mut ux2 = 0u32;
    let mut uy2 = 0u32;
    for m in &masks {
        ux1 = ux1.min(m.x);
        uy1 = uy1.min(m.y);
        ux2 = ux2.max(m.x + m.image.width());
        uy2 = uy2.max(m.y + m.image.height());
    }
    let uw = ux2 - ux1;
    let uh = uy2 - uy1;

    let mut merged = image::GrayImage::new(uw, uh);
    for m in &masks {
        let dx = m.x - ux1;
        let dy = m.y - uy1;
        for ly in 0..m.image.height() {
            for lx in 0..m.image.width() {
                if m.image.get_pixel(lx, ly).0[0] == 255 {
                    merged.put_pixel(dx + lx, dy + ly, image::Luma([255]));
                }
            }
        }
    }

    Some(LocalTextMask { x: ux1, y: uy1, image: merged })
}

fn placeholder_region() -> TextRegion {
    TextRegion {
        polygon: vec![],
        crop: image::DynamicImage::new_rgb8(1, 1),
        confidence: 0.0,
        mask: None,
    }
}
