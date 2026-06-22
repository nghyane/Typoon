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

/** Build one scan unit per page; halo is taken from the adjacent page. */
export function planPageScans(pages: readonly PageSource[], config: ScanConfig): readonly PageScanUnit[] {
  const sorted = [...pages].sort((a, b) => a.pageIndex - b.pageIndex)
  return sorted.map((page, i) => {
    const prev = sorted[i - 1] ?? null
    const next = sorted[i + 1] ?? null
    return {
      pageIndex: page.pageIndex,
      source: page.source,
      prevIndex: prev?.pageIndex ?? null,
      nextIndex: next?.pageIndex ?? null,
      haloTopPx: prev ? haloPx(page.source.height, config) : 0,
      haloBottomPx: next ? haloPx(page.source.height, config) : 0,
    }
  })
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
