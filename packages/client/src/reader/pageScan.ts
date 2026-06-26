// reader/pageScan.ts — plan one scan unit per page + DOM measurement.
//
// identity = pageIndex (stable across reflow). Halo heights come from config.

import type { ChapterContentLayout } from '../domain/chapterContent'
import type { PageScanUnit } from '../domain/pageScan'
import type { PageSize } from '../domain/source'
import type { ScanConfig } from './translationConfig'

export interface PageSource {
  readonly pageIndex: number
  readonly source: PageSize
}

/**
 * Build one scan unit per page. Halo (a strip of the adjacent page captured with
 * the core) lets a bubble that crosses a page seam be OCR'd whole — needed for
 * vertically-stitched webtoons. Paginated manga (e.g. Japanese tategaki) has no
 * seam-crossing text; there the halo only feeds neighbour-page pixels into the
 * OCR, which fuses across the boundary into one page-spanning block. Pass
 * `halo: false` for those sources to keep OCR strictly page-local.
 */
export function planPageScans(pages: readonly PageSource[], config: ScanConfig, halo = true): readonly PageScanUnit[] {
  const sorted = [...pages].sort((a, b) => a.pageIndex - b.pageIndex)
  return sorted.map((page, i) => {
    const prev = sorted[i - 1] ?? null
    const next = sorted[i + 1] ?? null
    return {
      pageIndex: page.pageIndex,
      source: page.source,
      prevIndex: prev?.pageIndex ?? null,
      nextIndex: next?.pageIndex ?? null,
      haloTopPx: halo && prev ? haloPx(page.source.height, config) : 0,
      haloBottomPx: halo && next ? haloPx(page.source.height, config) : 0,
    }
  })
}

/** Japanese is paginated tategaki — no seam-crossing bubbles, so no halo. */
export function sourceUsesHalo(sourceLanguage: string | null | undefined): boolean {
  return !/^ja\b/u.test((sourceLanguage ?? '').toLowerCase().trim())
}

function haloPx(pageHeight: number, config: ScanConfig): number {
  return Math.min(config.haloMaxPx, Math.round(pageHeight * config.haloRatio))
}

/** A measured page from the live reader DOM (for viewport ordering only). */
export interface MeasuredPage {
  readonly pageIndex: number
  readonly source: PageSize
  readonly domTop: number
  readonly domHeight: number
}

export function measuredPagesFromLayout(layout: ChapterContentLayout): readonly MeasuredPage[] {
  return layout.pages.map(page => ({
    pageIndex: page.pageIndex,
    source: page.sourceSize,
    domTop: page.contentRect.y,
    domHeight: page.contentRect.height,
  }))
}
