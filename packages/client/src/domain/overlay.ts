/**
 * Final output from the translation pipeline.
 *
 * PageOverlay is in **source/display page** coordinates (not prepared!).
 * The web overlay layer maps these to visual container coords.
 *
 * Background expansion (SafeMargins) is pre-computed during compose
 * so the renderer never needs raw image pixels.
 */

import type { TextPlacement } from './planning'
import type { TranslatedUnit } from './translation'
import type { PageSize } from './source'
import type { SafeMarginsDebug } from '../render/backgroundFit'

export interface PageOverlay {
  /** Source/display page index. */
  readonly pageIndex: number

  /** Raw pixel dimensions of the source image. */
  readonly pageSize: PageSize

  /** Text placements in source-image coordinates. */
  readonly placements: readonly TextPlacement[]

  /** Translated text for each placement. */
  readonly translations: readonly TranslatedUnit[]

  /**
   * Pre-computed background expansion data for each placement in
   * `placements`.  One-to-one by index.  The renderer uses this
   * instead of raw image pixels.
   */
  readonly placementMargins: readonly SafeMarginsDebug[]
}
