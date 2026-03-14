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

/// Maximum absolute vertical gap (px) between a line and a group bbox.
const MAX_VERTICAL_GAP_PX: f64 = 40.0;

/// Maximum absolute horizontal gap (px) between a line and a group bbox.
const MAX_HORIZONTAL_GAP_PX: f64 = 30.0;

/// Minimum horizontal overlap ratio (line vs group bbox) to allow merging.
const MIN_OVERLAP_RATIO: f64 = 0.4;

/// Vertical gap multiplier relative to the candidate line's own height.
const VGAP_HEIGHT_MULT: f64 = 1.5;

/// Maximum angle difference (degrees) between a line and a group's dominant
/// orientation. Prevents merging horizontal dialogue (~0°) with vertical
/// SFX (~90°) or heavily rotated text.
const MAX_ANGLE_DIFF_DEG: f64 = 30.0;

/// Minimum bubble bbox dimensions to filter noise.
const MIN_BUBBLE_W: f64 = 80.0;
const MIN_BUBBLE_H: f64 = 35.0;

/// Tracked bounding box + dominant angle for a group.
struct GroupBbox {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    /// Running sum of line angles (degrees) for averaging.
    angle_sum: f64,
    /// Number of lines contributing to angle_sum.
    angle_count: usize,
}

impl GroupBbox {
    fn from_line(polygon: &[[f64; 2]]) -> Self {
        let (x1, y1, x2, y2) = bbox(polygon);
        let angle = line_angle(polygon);
        Self { x1, y1, x2, y2, angle_sum: angle, angle_count: 1 }
    }

    fn expand(&mut self, polygon: &[[f64; 2]]) {
        let (x1, y1, x2, y2) = bbox(polygon);
        self.x1 = self.x1.min(x1);
        self.y1 = self.y1.min(y1);
        self.x2 = self.x2.max(x2);
        self.y2 = self.y2.max(y2);
        self.angle_sum += line_angle(polygon);
        self.angle_count += 1;
    }

    fn w(&self) -> f64 { self.x2 - self.x1 }

    /// Average angle of all lines in this group (degrees, 0°=horizontal).
    fn avg_angle(&self) -> f64 {
        if self.angle_count > 0 { self.angle_sum / self.angle_count as f64 } else { 0.0 }
    }

    /// Vertical gap between this group bbox and a line bbox.
    fn v_gap_to(&self, ly1: f64, ly2: f64) -> f64 {
        if ly2 < self.y1 { self.y1 - ly2 }
        else if ly1 > self.y2 { ly1 - self.y2 }
        else { 0.0 }
    }

    /// Horizontal gap between this group bbox and a line bbox.
    fn h_gap_to(&self, lx1: f64, lx2: f64) -> f64 {
        if lx2 < self.x1 { self.x1 - lx2 }
        else if lx1 > self.x2 { lx1 - self.x2 }
        else { 0.0 }
    }

    /// Horizontal overlap ratio: overlap / max(group_w, line_w).
    /// Using max_w prevents short lines from inflating the ratio.
    fn h_overlap_ratio(&self, lx1: f64, lx2: f64) -> f64 {
        let overlap = self.x2.min(lx2) - self.x1.max(lx1);
        if overlap <= 0.0 { return 0.0; }
        let max_w = self.w().max(lx2 - lx1);
        if max_w > 0.0 { overlap / max_w } else { 0.0 }
    }
}

/// Group PP-OCR text lines into bubble groups by spatial proximity.
///
/// Uses greedy group assignment instead of union-find to avoid transitive
/// merge bugs. Each line is assigned to the closest existing group by
/// checking distance to the group's accumulated bbox — not individual members.
///
/// Result sorted top-to-bottom for reading order.
pub fn group_lines(lines: Vec<TextRegion>) -> Vec<MergedBubble> {
    if lines.is_empty() {
        return Vec::new();
    }

    let mut lines = lines;
    lines.sort_by(|a, b| top_y(&a.polygon).partial_cmp(&top_y(&b.polygon)).unwrap());

    // Greedy group assignment
    let mut groups: Vec<(GroupBbox, Vec<usize>)> = Vec::new();

    for i in 0..lines.len() {
        let (lx1, ly1, lx2, ly2) = bbox(&lines[i].polygon);
        let lh = ly2 - ly1;
        let l_angle = line_angle(&lines[i].polygon);

        let mut best_group: Option<usize> = None;
        let mut best_vgap = f64::INFINITY;

        for (gi, (gb, _)) in groups.iter().enumerate() {
            // Orientation check: reject if angle differs too much from group
            let angle_diff = (l_angle - gb.avg_angle()).abs();
            if angle_diff > MAX_ANGLE_DIFF_DEG {
                continue;
            }

            let v_gap = gb.v_gap_to(ly1, ly2);
            let h_gap = gb.h_gap_to(lx1, lx2);

            // Absolute limits
            if v_gap > MAX_VERTICAL_GAP_PX || h_gap > MAX_HORIZONTAL_GAP_PX {
                continue;
            }

            // Relative vertical limit: use candidate line's own height
            if v_gap > lh * VGAP_HEIGHT_MULT {
                continue;
            }

            // Horizontal overlap with group bbox
            if gb.h_overlap_ratio(lx1, lx2) < MIN_OVERLAP_RATIO {
                continue;
            }

            // Pick closest group by vertical gap
            if v_gap < best_vgap {
                best_vgap = v_gap;
                best_group = Some(gi);
            }
        }

        if let Some(gi) = best_group {
            groups[gi].0.expand(&lines[i].polygon);
            groups[gi].1.push(i);
        } else {
            groups.push((GroupBbox::from_line(&lines[i].polygon), vec![i]));
        }
    }

    // Build MergedBubble results
    let mut result = Vec::new();
    for (_, indices) in groups {
        let mut group_lines: Vec<TextRegion> = indices.into_iter()
            .map(|i| std::mem::replace(&mut lines[i], placeholder_region()))
            .collect();

        sort_reading_order(&mut group_lines);

        let polygon = bounding_polygon(&group_lines);

        let (x1, y1, x2, y2) = bbox(&polygon);
        if (x2 - x1) < MIN_BUBBLE_W || (y2 - y1) < MIN_BUBBLE_H {
            continue;
        }

        // NOTE: SFX filtering in pipeline/mod.rs where OCR confidence is available.
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

    let metas: Vec<(f64, f64, f64)> = lines.iter().map(|l| {
        let (x1, y1, _, y2) = bbox(&l.polygon);
        let cy = (y1 + y2) / 2.0;
        let h = y2 - y1;
        (cy, h, x1)
    }).collect();

    let max_h = metas.iter().map(|m| m.1).fold(0.0_f64, f64::max);
    let row_tolerance = max_h * 0.5;

    let mut indices: Vec<usize> = (0..lines.len()).collect();
    indices.sort_by(|&a, &b| metas[a].0.partial_cmp(&metas[b].0).unwrap());

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

    for row in &mut rows {
        row.sort_by(|&a, &b| metas[a].2.partial_cmp(&metas[b].2).unwrap());
    }

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

/// Compute the orientation angle of a text line from its polygon (degrees).
///
/// For a rotated quad [TL, TR, BR, BL], the angle is atan2 of the top edge
/// (TL→TR). Returns 0° for horizontal, 90° for vertical.
/// For non-quad polygons, falls back to AABB aspect ratio estimation.
fn line_angle(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() >= 2 {
        // Use first edge (TL→TR) as the line's principal direction
        let dx = polygon[1][0] - polygon[0][0];
        let dy = polygon[1][1] - polygon[0][1];
        let angle = dy.atan2(dx).to_degrees().abs();
        // Normalize to [0, 90]: both 0° and 180° mean horizontal
        if angle > 90.0 { 180.0 - angle } else { angle }
    } else {
        0.0
    }
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

/// Composite per-line text masks into a single bubble mask (logical OR).
fn composite_masks(lines: &[TextRegion]) -> Option<LocalTextMask> {
    let masks: Vec<&LocalTextMask> = lines.iter().filter_map(|l| l.mask.as_ref()).collect();
    if masks.is_empty() {
        return None;
    }
    if masks.len() == 1 {
        return Some(masks[0].clone());
    }

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
