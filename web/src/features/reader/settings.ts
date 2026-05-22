// Reader settings — Kotatsu-style minimal surface.
//
// User-facing settings:
//   mode       4 modes (standard / rtl / vertical / webtoon)
//   pageWidth  desktop max width (px)
//
// Other behaviors use sensible defaults.

import { useMemo } from 'react'
import {
  useLocalSettings, useUpdateLocalSettings,
} from '@features/settings/local'


export type ReaderMode = 'pager' | 'strip'
/** Per-mode direction. Only meaningful for pager modes. */
export type ReaderDirection = 'ltr' | 'rtl' | 'ttb'

/** Combined reading style — what the user picks in the sheet. */
export type ReadingStyle = 'standard' | 'rtl' | 'vertical' | 'webtoon'


export interface ReaderSettings {
  /** Reading style — combined mode + direction. */
  style:     ReadingStyle
  pageWidth: number
}


export const READER_DEFAULTS: ReaderSettings = {
  style:     'rtl',     // Vietnamese manga audience defaults to RTL
  pageWidth: 1040,
}


export const PAGE_WIDTH_MIN = 600
export const PAGE_WIDTH_MAX = 1600


function clamp(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo
  return Math.min(hi, Math.max(lo, Math.round(n)))
}


/** Derive (mode, direction) from a high-level reading style. */
export function styleToLayout(style: ReadingStyle): {
  mode:      ReaderMode
  direction: ReaderDirection
} {
  switch (style) {
    case 'standard': return { mode: 'pager', direction: 'ltr' }
    case 'rtl':      return { mode: 'pager', direction: 'rtl' }
    case 'vertical': return { mode: 'pager', direction: 'ttb' }
    case 'webtoon':  return { mode: 'strip', direction: 'ttb' }
  }
}


export function useReaderSettings(): ReaderSettings {
  const q = useLocalSettings()
  return useMemo<ReaderSettings>(() => {
    const s = q.data
    if (!s) return READER_DEFAULTS
    // `reader_mode` stores the same string family but legacy values are
    // 'pager' | 'strip'. Migrate on read.
    const raw = (s.reader_mode ?? '') as string
    const style: ReadingStyle =
      raw === 'standard' || raw === 'rtl' || raw === 'vertical' || raw === 'webtoon'
        ? raw
        : raw === 'strip' ? 'webtoon'
        : raw === 'pager' ? 'rtl'
        :                   READER_DEFAULTS.style
    return {
      style,
      pageWidth: clamp(s.reader_page_width ?? READER_DEFAULTS.pageWidth, PAGE_WIDTH_MIN, PAGE_WIDTH_MAX),
    }
  }, [q.data])
}


export function usePatchReaderSettings() {
  const m = useUpdateLocalSettings()
  return (patch: Partial<ReaderSettings>) => {
    m.mutate({
      reader_mode:       patch.style as 'pager' | 'strip' | undefined,
      reader_page_width: patch.pageWidth !== undefined
        ? clamp(patch.pageWidth, PAGE_WIDTH_MIN, PAGE_WIDTH_MAX)
        : undefined,
    })
  }
}
