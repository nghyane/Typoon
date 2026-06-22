// domain/pageScan.ts — page-anchored translation contracts.
//
// Identity is the page index (reflow-invariant). Each page is OCR'd on a canvas
// = top halo (bottom strip of prev page) + the page itself (core) + bottom halo
// (top strip of next page). Ownership of a recognized block is decided by
// centroid-in-core; render target is decided by bbox geometry:
//   - bbox fully inside the page    → page surface (clipped per page)
//   - bbox crosses into next page   → seam-below bridge
//   - bbox crosses into prev page   → seam-above bridge
// A block whose centroid is outside the core belongs to the neighbor's scan and
// is dropped here, so every block is owned and rendered exactly once.

import type { BBox } from './geometry'
import type { TextPlacement } from './planning'
import type { PageSize } from './source'
import type { TranslatedUnit } from './translation'
import type { SafeMarginsDebug } from '../render/backgroundFit'

/** A stateless scan unit. identity === pageIndex. */
export interface PageScanUnit {
  readonly pageIndex: number
  readonly source: PageSize          // (sw, sh) of this page, in source px
  readonly prevIndex: number | null   // source page for the top halo
  readonly nextIndex: number | null   // source page for the bottom halo
  readonly haloTopPx: number          // halo height in this page's source px
  readonly haloBottomPx: number
}

export interface PlacementItem {
  readonly placement: TextPlacement
  readonly margin: SafeMarginsDebug
}

/**
 * A bubble that crosses the seam between two stacked pages. Rendered on a bridge
 * overlay that spans both page elements (not clipped to either). Coordinates are
 * seam-local: origin at the top of the top page, y increases downward through
 * the top page and into the bottom page's strip — i.e. the same space as the
 * owning page's capture canvas (minus halo offset), so no extra transform.
 */
export interface SeamOverlay {
  readonly topPageIndex: number
  readonly bottomPageIndex: number
  /** seam-local canvas size: [width, topPageSourceHeight + neighborHaloPx]. */
  readonly seamSize: readonly [number, number]
  /** y (seam-local px) where the bottom page begins (= top page source height). */
  readonly seamSplitY: number
  readonly items: readonly PlacementItem[]
  readonly translations: readonly TranslatedUnit[]
}

/** Per-page overlay: page-local items + optional seam bridges above/below. */
export interface ReaderPageOverlay {
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]   // = source
  readonly items: readonly PlacementItem[]
  readonly translations: readonly TranslatedUnit[]
  readonly seamBelow: SeamOverlay | null          // crosses into pageIndex + 1
  readonly seamAbove: SeamOverlay | null          // crosses into pageIndex - 1
}

export function emptyReaderPageOverlay(pageIndex: number, source: PageSize): ReaderPageOverlay {
  return {
    pageIndex,
    pageSize: [source.width, source.height],
    items: [],
    translations: [],
    seamBelow: null,
    seamAbove: null,
  }
}

export function bboxCentroidY(bbox: BBox): number {
  return (bbox[1] + bbox[3]) / 2
}

export function bboxCentroidX(bbox: BBox): number {
  return (bbox[0] + bbox[2]) / 2
}
