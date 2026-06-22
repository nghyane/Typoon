// reader/chunkCapture.ts — DOM layout measurement for page-scan planning.
// Stripped of chunk/strip capture (dead since page-anchored pipeline).

import type { ChapterContentLayout } from '../domain/chapterContent'
import type { SourcePageSize } from '../pipeline/chapterContent'

export interface VisibleContentRange {
  readonly top: number
  readonly bottom: number
  readonly center: number
}

/** Measure the live reader DOM into a chapter content layout. */
export function measureLayout(
  host: HTMLElement | null,
  knownSize: (pageIndex: number) => SourcePageSize | null,
): ChapterContentLayout | null {
  const container = host?.parentElement
  if (!host?.isConnected || !container) return null

  const containerRect = container.getBoundingClientRect()
  const pages = [...container.querySelectorAll<HTMLElement>('[data-page-index]')]
    .map(element => {
      const pageIndex = Number(element.dataset.pageIndex)
      const rect = element.getBoundingClientRect()
      const width = Math.max(1, rect.width)
      const height = Math.max(1, rect.height)
      return {
        pageIndex,
        sourceSize: knownSize(pageIndex) ?? { width, height },
        contentRect: {
          x: rect.left - containerRect.left,
          y: rect.top - containerRect.top,
          width,
          height,
        },
      }
    })
    .filter(page => Number.isFinite(page.pageIndex))
    .sort((a, b) => a.pageIndex - b.pageIndex)

  if (!pages.length) return null
  const contentWidth = Math.max(1, containerRect.width)
  const contentHeight = Math.max(
    1,
    container.scrollHeight,
    ...pages.map(page => page.contentRect.y + page.contentRect.height),
  )
  return { contentSize: { width: contentWidth, height: contentHeight }, pages }
}

export function visibleContentRange(
  host: HTMLElement,
  contentSize: { readonly width: number; readonly height: number },
  marginPx: number,
): VisibleContentRange {
  const rect = host.getBoundingClientRect()
  const scale = rect.width / Math.max(1, contentSize.width)
  if (!Number.isFinite(scale) || scale <= 0) return { top: 0, bottom: 0, center: 0 }
  const top = Math.max(0, (-rect.top - marginPx) / scale)
  const bottom = Math.min(contentSize.height, (window.innerHeight - rect.top + marginPx) / scale)
  return { top, bottom, center: (top + bottom) / 2 }
}
