use serde::{Deserialize, Serialize};

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

/// An ordered polygon (list of vertices).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon(Vec<[f64; 2]>);

impl Polygon {
    pub fn new(points: Vec<[f64; 2]>) -> Self {
        Self(points)
    }

    pub fn points(&self) -> &[[f64; 2]] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<[f64; 2]> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Axis-aligned bounding box.
    pub fn bbox(&self) -> Bbox {
        Bbox::from_polygon(&self.0)
    }

    /// Centroid (average of all vertices).
    pub fn center(&self) -> Point2 {
        let n = self.0.len().max(1) as f64;
        let sx: f64 = self.0.iter().map(|p| p[0]).sum();
        let sy: f64 = self.0.iter().map(|p| p[1]).sum();
        Point2 {
            x: sx / n,
            y: sy / n,
        }
    }
}

impl std::ops::Deref for Polygon {
    type Target = [[f64; 2]];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<[f64; 2]>> for Polygon {
    fn from(v: Vec<[f64; 2]>) -> Self {
        Self(v)
    }
}

impl From<Polygon> for Vec<[f64; 2]> {
    fn from(p: Polygon) -> Self {
        p.0
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl Bbox {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Compute from a slice of `[x, y]` points.
    pub fn from_polygon(polygon: &[[f64; 2]]) -> Self {
        let mut x1 = f64::INFINITY;
        let mut y1 = f64::INFINITY;
        let mut x2 = f64::NEG_INFINITY;
        let mut y2 = f64::NEG_INFINITY;
        for p in polygon {
            x1 = x1.min(p[0]);
            y1 = y1.min(p[1]);
            x2 = x2.max(p[0]);
            y2 = y2.max(p[1]);
        }
        Self { x1, y1, x2, y2 }
    }

    /// Returns (x1, y1, x2, y2) tuple for backward compatibility.
    pub fn as_tuple(&self) -> (f64, f64, f64, f64) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    pub fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn contains(&self, p: Point2) -> bool {
        p.x >= self.x1 && p.x <= self.x2 && p.y >= self.y1 && p.y <= self.y2
    }

    pub fn iou(&self, other: &Bbox) -> f64 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
}

/// Compute axis-aligned bounding box from a polygon slice.
///
/// Convenience free function — delegates to `Bbox::from_polygon`.
/// Used by modules that work with raw `&[[f64; 2]]` slices.
pub fn polygon_bbox(polygon: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    Bbox::from_polygon(polygon).as_tuple()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_from_polygon() {
        let poly = Polygon::new(vec![[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]);
        let bb = poly.bbox();
        assert_eq!(bb.x1, 10.0);
        assert_eq!(bb.y1, 20.0);
        assert_eq!(bb.x2, 50.0);
        assert_eq!(bb.y2, 80.0);
        assert_eq!(bb.width(), 40.0);
        assert_eq!(bb.height(), 60.0);
    }

    #[test]
    fn test_center() {
        let poly = Polygon::new(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        let c = poly.center();
        assert!((c.x - 5.0).abs() < 1e-9);
        assert!((c.y - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou() {
        let a = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let b = Bbox::new(5.0, 5.0, 15.0, 15.0);
        let iou = a.iou(&b);
        // intersection = 5×5 = 25, union = 100+100-25 = 175
        assert!((iou - 25.0 / 175.0).abs() < 1e-9);
    }

    #[test]
    fn test_polygon_bbox_compat() {
        let pts = vec![[1.0, 2.0], [5.0, 8.0]];
        let (x1, y1, x2, y2) = polygon_bbox(&pts);
        assert_eq!((x1, y1, x2, y2), (1.0, 2.0, 5.0, 8.0));
    }

    #[test]
    fn test_deref() {
        let poly = Polygon::new(vec![[1.0, 2.0], [3.0, 4.0]]);
        let slice: &[[f64; 2]] = &poly;
        assert_eq!(slice.len(), 2);
    }
}
