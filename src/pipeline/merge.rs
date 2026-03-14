use image::{DynamicImage, GenericImageView};

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

// ── Thresholds (cheap gates first, expensive last) ──

const MAX_VERTICAL_GAP_PX: f64 = 40.0;
const MAX_HORIZONTAL_GAP_PX: f64 = 30.0;
const MIN_OVERLAP_RATIO: f64 = 0.4;
const VGAP_HEIGHT_MULT: f64 = 1.5;
const MAX_ANGLE_DIFF_DEG: f64 = 30.0;
const MAX_HEIGHT_RATIO: f64 = 1.8;
/// If a new line's gap exceeds this multiple of the group's median inter-line
/// gap, it's likely from a different bubble.
const GAP_CONSISTENCY_MULT: f64 = 2.5;
/// Luminance threshold (0–255) for a pixel to be considered "dark" (border).
const BORDER_DARK_THRESHOLD: u8 = 100;
/// Fraction of scan-line width that must be dark to count as a border row.
const BORDER_ROW_RATIO: f64 = 0.25;

const MIN_BUBBLE_W: f64 = 80.0;
const MIN_BUBBLE_H: f64 = 35.0;

// ── Group state ──

struct GroupState {
    // Bbox
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    // Running averages
    angle_sum: f64,
    height_sum: f64,
    line_count: usize,
    // Sorted bottom-y of each line (for nearest-line gap + gap consistency)
    line_bottoms: Vec<f64>,
    // Sorted inter-line gaps for consistency check
    gaps: Vec<f64>,
}

impl GroupState {
    fn from_line(polygon: &[[f64; 2]]) -> Self {
        let (x1, y1, x2, y2) = bbox(polygon);
        Self {
            x1, y1, x2, y2,
            angle_sum: line_angle(polygon),
            height_sum: y2 - y1,
            line_count: 1,
            line_bottoms: vec![y2],
            gaps: Vec::new(),
        }
    }

    fn add_line(&mut self, polygon: &[[f64; 2]], v_gap: f64) {
        let (x1, y1, x2, y2) = bbox(polygon);
        self.x1 = self.x1.min(x1);
        self.y1 = self.y1.min(y1);
        self.x2 = self.x2.max(x2);
        self.y2 = self.y2.max(y2);
        self.angle_sum += line_angle(polygon);
        self.height_sum += y2 - y1;
        self.line_count += 1;
        self.line_bottoms.push(y2);
        if v_gap > 0.0 {
            self.gaps.push(v_gap);
            self.gaps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    fn w(&self) -> f64 { self.x2 - self.x1 }

    fn avg_angle(&self) -> f64 {
        self.angle_sum / self.line_count.max(1) as f64
    }

    fn avg_height(&self) -> f64 {
        self.height_sum / self.line_count.max(1) as f64
    }

    fn median_gap(&self) -> Option<f64> {
        if self.gaps.is_empty() { return None; }
        Some(self.gaps[self.gaps.len() / 2])
    }

    fn v_gap_to(&self, ly1: f64, ly2: f64) -> f64 {
        if ly2 < self.y1 { self.y1 - ly2 }
        else if ly1 > self.y2 { ly1 - self.y2 }
        else { 0.0 }
    }

    fn h_gap_to(&self, lx1: f64, lx2: f64) -> f64 {
        if lx2 < self.x1 { self.x1 - lx2 }
        else if lx1 > self.x2 { lx1 - self.x2 }
        else { 0.0 }
    }

    fn h_overlap_ratio(&self, lx1: f64, lx2: f64) -> f64 {
        let overlap = self.x2.min(lx2) - self.x1.max(lx1);
        if overlap <= 0.0 { return 0.0; }
        let max_w = self.w().max(lx2 - lx1);
        if max_w > 0.0 { overlap / max_w } else { 0.0 }
    }
}

// ── Border detection ──

/// Check if there's a visible bubble border between two vertical regions.
///
/// Samples horizontal scan-lines in the gap between `upper_y2` and `lower_y1`,
/// within the horizontal overlap `[x_left, x_right]`.
/// A scan-line is a "border row" if ≥ 25% of its pixels are dark (luminance < 100).
/// Returns true if any border row is found.
fn has_border_between(
    img: &DynamicImage,
    upper_y2: f64,
    lower_y1: f64,
    x_left: f64,
    x_right: f64,
) -> bool {
    let gap_top = upper_y2.ceil() as u32;
    let gap_bot = lower_y1.floor() as u32;
    if gap_top >= gap_bot || x_left >= x_right {
        return false;
    }

    let (iw, ih) = img.dimensions();
    let xl = (x_left.floor() as u32).min(iw.saturating_sub(1));
    let xr = (x_right.ceil() as u32).min(iw);
    let scan_w = xr.saturating_sub(xl);
    if scan_w == 0 {
        return false;
    }

    for sy in gap_top..gap_bot.min(ih) {
        let mut dark_count = 0u32;
        for sx in xl..xr {
            let p = img.get_pixel(sx, sy);
            let lum = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000;
            if lum < BORDER_DARK_THRESHOLD as u32 {
                dark_count += 1;
            }
        }
        if dark_count as f64 / scan_w as f64 >= BORDER_ROW_RATIO {
            return true;
        }
    }
    false
}

// ── Main grouping ──

/// Group PP-OCR text lines into bubble groups by spatial proximity + visual
/// border detection.
///
/// Gates (ordered cheap → expensive, early-exit):
/// 1. Angle compatibility
/// 2. Height (font size) compatibility
/// 3. Spatial proximity (gap, overlap)
/// 4. Gap consistency (new gap vs group's median gap)
/// 5. Border detection (sample image pixels between lines)
pub fn group_lines(lines: Vec<TextRegion>, img: &DynamicImage) -> Vec<MergedBubble> {
    if lines.is_empty() {
        return Vec::new();
    }

    let mut lines = lines;
    lines.sort_by(|a, b| top_y(&a.polygon).partial_cmp(&top_y(&b.polygon)).unwrap());

    let mut groups: Vec<(GroupState, Vec<usize>)> = Vec::new();

    for i in 0..lines.len() {
        let (lx1, ly1, lx2, ly2) = bbox(&lines[i].polygon);
        let lh = ly2 - ly1;
        let l_angle = line_angle(&lines[i].polygon);

        let mut best_group: Option<usize> = None;
        let mut best_vgap = f64::INFINITY;

        for (gi, (gs, _)) in groups.iter().enumerate() {
            // Gate 1: orientation
            if (l_angle - gs.avg_angle()).abs() > MAX_ANGLE_DIFF_DEG {
                continue;
            }

            // Gate 2: font size
            let ratio = lh.max(gs.avg_height()) / lh.min(gs.avg_height()).max(1.0);
            if ratio > MAX_HEIGHT_RATIO {
                continue;
            }

            // Gate 3: spatial proximity
            let v_gap = gs.v_gap_to(ly1, ly2);
            let h_gap = gs.h_gap_to(lx1, lx2);
            if v_gap > MAX_VERTICAL_GAP_PX || h_gap > MAX_HORIZONTAL_GAP_PX {
                continue;
            }
            if v_gap > lh * VGAP_HEIGHT_MULT {
                continue;
            }
            if gs.h_overlap_ratio(lx1, lx2) < MIN_OVERLAP_RATIO {
                continue;
            }

            // Gate 4: gap consistency
            if let Some(median_gap) = gs.median_gap() {
                if v_gap > median_gap * GAP_CONSISTENCY_MULT {
                    continue;
                }
            }

            // Gate 5: border detection — check if there's a visible border
            // between the group's bottom and this line's top.
            let overlap_x1 = gs.x1.max(lx1);
            let overlap_x2 = gs.x2.min(lx2);
            if v_gap > 0.0 && has_border_between(img, gs.y2, ly1, overlap_x1, overlap_x2) {
                continue;
            }

            if v_gap < best_vgap {
                best_vgap = v_gap;
                best_group = Some(gi);
            }
        }

        if let Some(gi) = best_group {
            groups[gi].0.add_line(&lines[i].polygon, best_vgap);
            groups[gi].1.push(i);
        } else {
            groups.push((GroupState::from_line(&lines[i].polygon), vec![i]));
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

fn line_angle(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() >= 2 {
        let dx = polygon[1][0] - polygon[0][0];
        let dy = polygon[1][1] - polygon[0][1];
        let angle = dy.atan2(dx).to_degrees().abs();
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
