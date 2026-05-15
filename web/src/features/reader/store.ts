// Persistent reader settings. Two scopes:
//
//   - Global per device: pageWidth, imageFit, pageGap, stripMargin,
//     behavior flags. Shared across every work / chapter.
//   - Per-work: reading direction. Manga reads RTL, manhua reads
//     LTR, webtoon reads TTB — bound to the work, not the device.
//
// Zustand + persist middleware so settings survive reload / new
// tabs. Same pattern as `store/sidebar.ts`. Mutations are field-
// level setters (no granular actions) because reader settings are
// dense and most edits come from a single Settings sheet — exposing
// per-field actions would balloon the surface for no clarity win.

import { create } from 'zustand'
import { persist } from 'zustand/middleware'


export type Direction = 'ltr' | 'rtl' | 'ttb'
export type ImageFit  = 'width' | 'height' | 'free'


/** What version of a chapter the user prefers to read on this Work.
 *  Sticky across chapters so a user who picks "AI VI từ EN" on Ch.10
 *  keeps reading from EN on Ch.11, 12, … without re-picking.
 *
 *  Granularity is `(kind, lang, sourceLang?)` \u2014 NOT material_id.
 *  Reasoning: a user thinks "I'm reading the AI VI from English",
 *  not "AI VI from English MangaDex material 42". When Ch.99 doesn't
 *  have that exact MangaDex material but has another English raw,
 *  the pref still matches and the new material's draft is used. */
export interface SourcePreference {
  /** 'translation' \u2014 AI render; 'raw' \u2014 native scanlation. */
  kind:        'translation' | 'raw'
  /** For translation: target_lang (what the user reads in).
   *  For raw:         the raw's own lang. */
  lang:        string
  /** Translation only \u2014 the source_lang the draft was rendered from
   *  (`v.sourceLang`). Pref matches translations whose `sourceLang`
   *  equals this; null \u2192 any source_lang acceptable. */
  sourceLang?: string | null
}


export interface ReaderSettingsState {
  // Global (per device)
  pageWidth:     number   // px, clamped at runtime: 600..1600
  imageFit:      ImageFit
  pageGap:       number   // px gap between pages in TTB
  stripMargin:   number   // px top/bottom of strip container
  clickTurnPage: boolean
  swipeGestures: boolean
  preloadAhead:  number   // count of pages
  resumePosition: boolean

  // Per-work overrides
  directionByWork:  Record<string, Direction>
  sourcePrefByWork: Record<string, SourcePreference>

  // Mutations
  setPageWidth:    (px: number) => void
  setImageFit:     (fit: ImageFit) => void
  setPageGap:      (px: number) => void
  setStripMargin:  (px: number) => void
  setClickTurnPage:(on: boolean) => void
  setSwipeGestures:(on: boolean) => void
  setPreloadAhead: (n: number) => void
  setResumePosition: (on: boolean) => void
  setDirection:    (workId: number, dir: Direction) => void
  setSourcePref:   (workId: number, pref: SourcePreference | null) => void
}


// Bounds + defaults centralised so the slider UI doesn't drift from
// the store. The setters clamp aggressively rather than trusting
// the caller — a runaway slider drag must never park `pageWidth`
// at 0 or `Infinity`.
export const PAGE_WIDTH_MIN  = 600
export const PAGE_WIDTH_MAX  = 1600
export const PAGE_GAP_MAX    = 32
export const STRIP_MARGIN_MAX = 32
export const PRELOAD_AHEAD_MAX = 8


function clamp(n: number, lo: number, hi: number) {
  if (!Number.isFinite(n)) return lo
  return Math.min(hi, Math.max(lo, Math.round(n)))
}


export const useReaderSettings = create<ReaderSettingsState>()(
  persist(
    (set) => ({
      pageWidth:     1040,
      imageFit:      'width',
      pageGap:       8,
      stripMargin:   16,
      clickTurnPage: true,
      swipeGestures: true,
      preloadAhead:  3,
      resumePosition: true,

      directionByWork:  {},
      sourcePrefByWork: {},

      setPageWidth:   (px)  => set({ pageWidth:   clamp(px, PAGE_WIDTH_MIN, PAGE_WIDTH_MAX) }),
      setImageFit:    (fit) => set({ imageFit:    fit }),
      setPageGap:     (px)  => set({ pageGap:     clamp(px, 0, PAGE_GAP_MAX) }),
      setStripMargin: (px)  => set({ stripMargin: clamp(px, 0, STRIP_MARGIN_MAX) }),
      setClickTurnPage: (on) => set({ clickTurnPage: on }),
      setSwipeGestures: (on) => set({ swipeGestures: on }),
      setPreloadAhead:  (n)  => set({ preloadAhead: clamp(n, 0, PRELOAD_AHEAD_MAX) }),
      setResumePosition:(on) => set({ resumePosition: on }),
      setDirection:     (workId, dir) => set((s) => ({
        directionByWork: { ...s.directionByWork, [String(workId)]: dir },
      })),
      setSourcePref:    (workId, pref) => set((s) => {
        const key = String(workId)
        const next = { ...s.sourcePrefByWork }
        if (pref === null) delete next[key]
        else next[key] = pref
        return { sourcePrefByWork: next }
      }),
    }),
    {
      name: 'typoon_reader_settings',
      // Migrate older saved blobs if the shape changes. Keep
      // numeric defaults so a stored `undefined` from a previous
      // version doesn't break the slider.
      version: 1,
    },
  ),
)


/** Read the effective direction for a work. Falls back to TTB
 *  (webtoon / continuous strip) — the most common default for
 *  Korean/Chinese webcomics and a safe pick when the work doesn't
 *  yet have a user-chosen direction. */
export function directionFor(
  state: Pick<ReaderSettingsState, 'directionByWork'>,
  workId: number,
): Direction {
  return state.directionByWork[String(workId)] ?? 'ttb'
}


/** Read the user's preferred source for a work. `null` when none —
 *  the reader falls back to `pickReadable`'s default priority. */
export function sourcePrefFor(
  state: Pick<ReaderSettingsState, 'sourcePrefByWork'>,
  workId: number,
): SourcePreference | null {
  return state.sourcePrefByWork[String(workId)] ?? null
}
