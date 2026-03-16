use image::{DynamicImage, GenericImageView, GrayImage};

use crate::vision::detection::{LocalTextMask, TextRegion};

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

// ── Thresholds ──
// Ratio-based thresholds (resolution-independent)
const MIN_OVERLAP_RATIO: f64 = 0.4;
const MAX_ANGLE_DIFF_DEG: f64 = 30.0;
const MAX_HEIGHT_RATIO: f64 = 1.8;
const GAP_CONSISTENCY_MULT: f64 = 2.5;
/// Minimum luminance drop from the local background to detect a border edge.
const BORDER_CONTRAST_DROP: u32 = 40;
/// Fraction of scan-line width that must show contrast-drop to count as border.
const BORDER_ROW_RATIO: f64 = 0.25;
/// Mean DB probability (0–255) in the gap between two lines above which
/// we consider them connected by text (e.g. underline, continuation).
const PROB_BRIDGE_THRESHOLD: u8 = 38;

/// Adaptive thresholds derived from the actual detected text lines on this page.
struct PageStats {
    /// Max vertical gap: lines farther apart than this are separate bubbles.
    /// Derived from median line height — typical intra-bubble line spacing
    /// is well under 1.5× the line height.
    max_v_gap: f64,
    /// Max horizontal gap between a line and a group.
    max_h_gap: f64,
    /// Minimum bubble width to keep (filters noise).
    min_bubble_w: f64,
    /// Minimum bubble height to keep.
    min_bubble_h: f64,
}

impl PageStats {
    fn from_lines(lines: &[TextRegion]) -> Self {
        let mut heights: Vec<f64> = lines
            .iter()
            .map(|l| {
                let (_, y1, _, y2) = bbox(&l.polygon);
                y2 - y1
            })
            .filter(|&h| h > 1.0)
            .collect();
        heights.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let median_h = if heights.is_empty() {
            20.0 // degenerate fallback
        } else {
            heights[heights.len() / 2]
        };

        Self {
            max_v_gap: median_h * 1.5,
            max_h_gap: median_h * 1.0,
            min_bubble_w: median_h * 2.5,
            min_bubble_h: median_h * 1.0,
        }
    }
}

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
            x1,
            y1,
            x2,
            y2,
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

    fn w(&self) -> f64 {
        self.x2 - self.x1
    }

    fn avg_angle(&self) -> f64 {
        self.angle_sum / self.line_count.max(1) as f64
    }

    fn avg_height(&self) -> f64 {
        self.height_sum / self.line_count.max(1) as f64
    }

    fn median_gap(&self) -> Option<f64> {
        if self.gaps.is_empty() {
            return None;
        }
        Some(self.gaps[self.gaps.len() / 2])
    }

    fn v_gap_to(&self, ly1: f64, ly2: f64) -> f64 {
        if ly2 < self.y1 {
            self.y1 - ly2
        } else if ly1 > self.y2 {
            ly1 - self.y2
        } else {
            0.0
        }
    }

    fn h_gap_to(&self, lx1: f64, lx2: f64) -> f64 {
        if lx2 < self.x1 {
            self.x1 - lx2
        } else if lx1 > self.x2 {
            lx1 - self.x2
        } else {
            0.0
        }
    }

    fn h_overlap_ratio(&self, lx1: f64, lx2: f64) -> f64 {
        let overlap = self.x2.min(lx2) - self.x1.max(lx1);
        if overlap <= 0.0 {
            return 0.0;
        }
        // Use min(widths): a short centered line (e.g. "MÀ, ĐÚNG KHÔNG?")
        // inside a wide group should have high ratio. overlap/max penalizes
        // width disparity; overlap/min asks "is the shorter one contained?"
        let min_w = self.w().min(lx2 - lx1);
        if min_w > 0.0 { overlap / min_w } else { 0.0 }
    }
}

// ── Border detection ──

/// Check if there's a visible bubble border between two vertical regions.
///
/// Instead of absolute darkness, detects **contrast edges**: pixels that are
/// significantly darker than the local background (median luminance of the
/// first/last rows of the gap region). This works on both light and dark
/// backgrounds — a border is always a *relative* contrast feature.
///
/// Returns true if any scan-line in the gap has ≥ 25% of pixels showing a
/// luminance drop ≥ BORDER_CONTRAST_DROP from the local background.
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
    if scan_w < 3 {
        return false;
    }
    let gap_bot = gap_bot.min(ih);

    // Compute local background luminance: median of the top and bottom edge
    // rows of the gap (adjacent to the text lines, where bubble interior is).
    let bg_lum = {
        let mut edge_lums = Vec::new();
        for &sy in &[gap_top, gap_bot.saturating_sub(1)] {
            if sy < ih {
                for sx in (xl..xr).step_by(2) {
                    edge_lums.push(pixel_lum(img, sx, sy));
                }
            }
        }
        if edge_lums.is_empty() {
            return false;
        }
        edge_lums.sort_unstable();
        edge_lums[edge_lums.len() / 2]
    };

    // Scan each row: count pixels with luminance significantly below local bg
    for sy in gap_top..gap_bot {
        let mut contrast_count = 0u32;
        for sx in xl..xr {
            let lum = pixel_lum(img, sx, sy);
            if bg_lum >= BORDER_CONTRAST_DROP && lum < bg_lum - BORDER_CONTRAST_DROP {
                contrast_count += 1;
            }
        }
        if contrast_count as f64 / scan_w as f64 >= BORDER_ROW_RATIO {
            return true;
        }
    }
    false
}

/// Check if the DB probability map shows text signal in the gap between two
/// vertical regions. High mean probability = text bridge (same text block).
///
/// This is a positive signal (encourages merging), complementary to
/// `has_border_between` which is a negative signal (blocks merging).
fn has_prob_bridge(
    prob_img: &GrayImage,
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
    let (pw, ph) = prob_img.dimensions();
    let xl = (x_left.floor() as u32).min(pw.saturating_sub(1));
    let xr = (x_right.ceil() as u32).min(pw);
    let gap_bot = gap_bot.min(ph);
    if xr <= xl {
        return false;
    }

    let mut sum = 0u64;
    let mut count = 0u64;
    for sy in gap_top..gap_bot {
        for sx in xl..xr {
            sum += prob_img.get_pixel(sx, sy).0[0] as u64;
            count += 1;
        }
    }
    if count == 0 {
        return false;
    }
    (sum / count) as u8 >= PROB_BRIDGE_THRESHOLD
}

/// BT.601 luminance of a pixel.
fn pixel_lum(img: &DynamicImage, x: u32, y: u32) -> u32 {
    let p = img.get_pixel(x, y);
    (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000
}

// ── Main grouping ──

/// Group PP-OCR text lines into bubble groups by spatial proximity + visual
/// border detection + text probability bridging.
///
/// Gates (ordered cheap → expensive, early-exit):
/// 1. Angle compatibility
/// 2. Height (font size) compatibility
/// 3. Spatial proximity (gap, overlap)
/// 4. Gap consistency (new gap vs group's median gap)
/// 5. Border detection (sample image pixels between lines)
/// 6. Prob bridge (DB probability in gap — positive signal overrides gate 5)
pub fn group_lines(
    lines: Vec<TextRegion>,
    img: &DynamicImage,
    prob_image: Option<&GrayImage>,
) -> Vec<MergedBubble> {
    if lines.is_empty() {
        return Vec::new();
    }

    let mut lines = lines;
    lines.sort_by(|a, b| top_y(&a.polygon).partial_cmp(&top_y(&b.polygon)).unwrap());

    let stats = PageStats::from_lines(&lines);

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

            // Gate 3: spatial proximity (adaptive to page text size)
            let v_gap = gs.v_gap_to(ly1, ly2);
            let h_gap = gs.h_gap_to(lx1, lx2);
            if v_gap > stats.max_v_gap || h_gap > stats.max_h_gap {
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

            // Gate 5+6: border detection vs prob bridge.
            let overlap_x1 = gs.x1.max(lx1);
            let overlap_x2 = gs.x2.min(lx2);
            if v_gap > 0.0 && has_border_between(img, gs.y2, ly1, overlap_x1, overlap_x2) {
                let bridged = prob_image
                    .is_some_and(|pi| has_prob_bridge(pi, gs.y2, ly1, overlap_x1, overlap_x2));
                if !bridged {
                    continue;
                }
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
        let mut group_lines: Vec<TextRegion> = indices
            .into_iter()
            .map(|i| std::mem::replace(&mut lines[i], placeholder_region()))
            .collect();

        sort_reading_order(&mut group_lines);

        let polygon = bounding_polygon(&group_lines);

        let (x1, y1, x2, y2) = bbox(&polygon);
        if (x2 - x1) < stats.min_bubble_w || (y2 - y1) < stats.min_bubble_h {
            continue;
        }

        let confidence = group_lines
            .iter()
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

    let metas: Vec<(f64, f64, f64)> = lines
        .iter()
        .map(|l| {
            let (x1, y1, _, y2) = bbox(&l.polygon);
            let cy = (y1 + y2) / 2.0;
            let h = y2 - y1;
            (cy, h, x1)
        })
        .collect();

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
    let mut sorted: Vec<TextRegion> = sorted_indices
        .into_iter()
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

    // Dilate after merge to cover anti-aliased edges between lines.
    // Radius scales with average line height (~8% of line height, min 2px).
    let avg_line_h = masks.iter().map(|m| m.image.height()).sum::<u32>() / masks.len() as u32;
    let pad_r = (avg_line_h / 8).max(2);
    let merged = crate::vision::detection::dilate_mask(&merged, pad_r);

    Some(LocalTextMask {
        x: ux1,
        y: uy1,
        image: merged,
    })
}

fn placeholder_region() -> TextRegion {
    TextRegion {
        polygon: vec![],
        crop: image::DynamicImage::new_rgb8(1, 1),
        confidence: 0.0,
        mask: None,
    }
}
