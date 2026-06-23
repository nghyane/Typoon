export type BBox = readonly [number, number, number, number];
export type Point = readonly [number, number];
export type Polygon = readonly Point[];
/**
 * Oriented (rotated) rectangle in page-pixel space.  Unlike BBox, `w`/`h` are
 * the true unrotated extents, so a tilted text line keeps its real font size
 * instead of the inflated axis-aligned height (`w·sinθ + h·cosθ`).  This is the
 * geometry the OCR engine actually returns; keeping it avoids lossy measurement
 * downstream.
 */
export interface OrientedBox {
    readonly cx: number;
    readonly cy: number;
    readonly w: number;
    readonly h: number;
    readonly rotationDeg: number;
}
export declare function orientedFromBBox(bbox: BBox, rotationDeg?: number): OrientedBox;
/** Overlap ratio of two 1-D intervals (centers separated by `shift`), normalized by the smaller extent. */
export declare function overlapRatio1D(shift: number, extentA: number, extentB: number): number;
export interface OrientedPairFrame {
    /** Gap between box edges along the line-stacking axis (perpendicular to reading). */
    readonly primaryGapPx: number;
    /** Overlap ratio along the reading axis. */
    readonly secondaryOverlap: number;
    /** Center shift along the reading axis. */
    readonly centerShiftPx: number;
    /** Larger reading-axis extent of the pair. */
    readonly maxSecondarySpan: number;
    /** Font size = extent perpendicular to reading. */
    readonly fontPx: number;
}
/**
 * Measure a pair of oriented boxes in their shared rotated frame.  For
 * horizontal text the reading axis is the box local-x (`u`) and lines stack
 * along local-y (`v`); for vertical text the axes swap.  Measuring here instead
 * of on axis-aligned bboxes keeps gaps/overlaps correct when text is tilted.
 */
export declare function orientedPairFrame(a: OrientedBox, b: OrientedBox, direction: 'horizontal' | 'vertical'): OrientedPairFrame;
export declare function area(b: BBox): number;
export declare function center(b: BBox): Point;
export declare function containsCenter(outer: BBox, inner: BBox): boolean;
export declare function containment(inner: BBox, outer: BBox): number;
export declare function iou(a: BBox, b: BBox): number;
export declare function union(bboxes: readonly BBox[]): BBox;
export declare function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox;
export declare function bboxToPolygon(b: BBox): Polygon;
export declare function polygonBBox(polygon: Polygon, pageSize: readonly [number, number]): BBox;
export declare function inscribedEllipse(bbox: BBox, vertices?: number): Polygon;
