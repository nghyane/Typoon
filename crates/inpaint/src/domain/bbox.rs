// SPDX-License-Identifier: GPL-3.0-or-later
//! Axis-aligned bounding box in page pixel coordinates.

use anyhow::{anyhow, Result};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use std::fmt;

/// Axis-aligned bounding box `[x1, y1, x2, y2]` in page pixels.
///
/// Invariants enforced at construction:
/// - `x1 < x2`  (non-degenerate horizontal)
/// - `y1 < y2`  (non-degenerate vertical)
/// - all values ≥ 0
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl BBox {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Result<Self> {
        if x1 >= x2 || y1 >= y2 {
            return Err(anyhow!("degenerate bbox ({x1},{y1},{x2},{y2})"));
        }
        Ok(Self { x1, y1, x2, y2 })
    }

    /// Shorter of width / height — used for close-radius computation.
    pub fn short_edge(&self) -> i32 {
        (self.x2 - self.x1).min(self.y2 - self.y1)
    }

    pub fn width(&self)  -> i32 { self.x2 - self.x1 }
    pub fn height(&self) -> i32 { self.y2 - self.y1 }
    pub fn area(&self)   -> i64 { self.width() as i64 * self.height() as i64 }

    /// Centre (float for precision).
    pub fn centre(&self) -> (f32, f32) {
        (
            (self.x1 + self.x2) as f32 / 2.0,
            (self.y1 + self.y2) as f32 / 2.0,
        )
    }

    /// Clamp to a page boundary `[0, w) × [0, h)`.
    pub fn clamp(&self, w: i32, h: i32) -> Self {
        Self {
            x1: self.x1.max(0),
            y1: self.y1.max(0),
            x2: self.x2.min(w),
            y2: self.y2.min(h),
        }
    }

    /// Expand by `r` pixels in all directions, clamped to page.
    pub fn expand(&self, r: i32, w: i32, h: i32) -> Self {
        Self {
            x1: (self.x1 - r).max(0),
            y1: (self.y1 - r).max(0),
            x2: (self.x2 + r).min(w),
            y2: (self.y2 + r).min(h),
        }
    }

    /// L∞ distance between two boxes (0 when touching or overlapping).
    pub fn distance_to(&self, other: &BBox) -> i32 {
        let dx = if self.x2 < other.x1 { other.x1 - self.x2 }
                 else if other.x2 < self.x1 { self.x1 - other.x2 }
                 else { 0 };
        let dy = if self.y2 < other.y1 { other.y1 - self.y2 }
                 else if other.y2 < self.y1 { self.y1 - other.y2 }
                 else { 0 };
        dx.max(dy)
    }

    pub fn union(&self, other: &BBox) -> BBox {
        BBox {
            x1: self.x1.min(other.x1), y1: self.y1.min(other.y1),
            x2: self.x2.max(other.x2), y2: self.y2.max(other.y2),
        }
    }
}

// ── Deserialize from `[x1, y1, x2, y2]` msgpack array, validated ─────────

struct BBoxVisitor;

impl<'de> Visitor<'de> for BBoxVisitor {
    type Value = BBox;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[x1, y1, x2, y2] integer array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let x1 = seq.next_element::<i32>()?.ok_or_else(|| de::Error::custom("missing x1"))?;
        let y1 = seq.next_element::<i32>()?.ok_or_else(|| de::Error::custom("missing y1"))?;
        let x2 = seq.next_element::<i32>()?.ok_or_else(|| de::Error::custom("missing x2"))?;
        let y2 = seq.next_element::<i32>()?.ok_or_else(|| de::Error::custom("missing y2"))?;
        BBox::new(x1, y1, x2, y2).map_err(de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for BBox {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_seq(BBoxVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_bbox() {
        let b = BBox::new(10, 20, 100, 200).unwrap();
        assert_eq!(b.short_edge(), 90);
        assert_eq!(b.width(), 90);
        assert_eq!(b.height(), 180);
    }

    #[test]
    fn degenerate_bbox_rejected() {
        assert!(BBox::new(10, 10, 10, 50).is_err());
        assert!(BBox::new(10, 10, 50, 10).is_err());
    }

    #[test]
    fn distance() {
        let a = BBox::new(0, 0, 10, 10).unwrap();
        let b = BBox::new(15, 0, 25, 10).unwrap();
        assert_eq!(a.distance_to(&b), 5);
        assert_eq!(b.distance_to(&a), 5);

        let c = BBox::new(5, 0, 20, 10).unwrap();
        assert_eq!(a.distance_to(&c), 0); // overlapping
    }
}
