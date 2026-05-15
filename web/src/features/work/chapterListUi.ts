// chapterListUi — persisted per-work UI state for the chapter list.
//
// `WorkChapterList` lives inside the work route. Each time the user
// navigates away (into the reader, onto another work) the component
// unmounts and any `useState` resets — filter pick / sort direction
// would be forgotten. Persisting them per-`workId` in a small zustand
// store survives unmounts AND tab reloads, so a reader who's been
// reading EN raws on One Piece keeps that filter pinned the next
// session.
//
// We persist only the pieces with cross-session intent:
//
//   • activeLang  — null = "Tất cả"; user's explicit pick survives
//                   navigation. Absence of an entry means "default
//                   policy applies" (component picks target_lang
//                   when reachable, else null).
//   • sortBy      — 'newest' / 'oldest' direction toggle.
//
// Search query `q` stays local — it's a transient typing affordance,
// not a long-lived intent.
//
// Cache size: 100 most-recent works. Each entry is ~20 bytes so the
// cap is purely defensive against a user browsing thousands of works
// over months. LRU eviction keyed by last-write timestamp.

import { create } from 'zustand'
import { useShallow } from 'zustand/react/shallow'
import { persist, createJSONStorage } from 'zustand/middleware'


export type ChapterSort = 'newest' | 'oldest'


export interface WorkListUiState {
  activeLang: string | null
  sortBy:     ChapterSort
  /** Last-write timestamp (epoch ms). Drives LRU eviction. */
  touchedAt:  number
}


interface Store {
  /** Per-work UI state. Keyed by `workId` stringified. */
  byWork: Record<string, WorkListUiState>

  setActiveLang: (workId: number, lang: string | null) => void
  setSortBy:     (workId: number, sortBy: ChapterSort) => void
}


const MAX_ENTRIES = 100


/** Default state seed used whenever we write to a previously-unknown
 *  workId. `sortBy` defaults to 'newest' (the original component
 *  default); activeLang is `null` placeholder — callers that haven't
 *  written an explicit pick get the null/sentinel back. */
function seed(): WorkListUiState {
  return { activeLang: null, sortBy: 'newest', touchedAt: Date.now() }
}


/** LRU eviction — when the entry count exceeds MAX_ENTRIES, drop the
 *  oldest `touchedAt` until under the cap. Operates on a plain object
 *  so the caller can return the result directly into `set`. */
function evict(byWork: Record<string, WorkListUiState>): Record<string, WorkListUiState> {
  const keys = Object.keys(byWork)
  if (keys.length <= MAX_ENTRIES) return byWork
  const sorted = keys
    .map((k) => [k, byWork[k]!.touchedAt] as const)
    .sort((a, b) => a[1] - b[1])
  const drop = sorted.slice(0, keys.length - MAX_ENTRIES).map(([k]) => k)
  if (drop.length === 0) return byWork
  const next = { ...byWork }
  for (const k of drop) delete next[k]
  return next
}


export const useChapterListUi = create<Store>()(
  persist(
    (set) => ({
      byWork: {},

      setActiveLang: (workId, lang) => set((s) => {
        const key  = String(workId)
        const prev = s.byWork[key] ?? seed()
        const next = { ...prev, activeLang: lang, touchedAt: Date.now() }
        return { byWork: evict({ ...s.byWork, [key]: next }) }
      }),

      setSortBy: (workId, sortBy) => set((s) => {
        const key  = String(workId)
        const prev = s.byWork[key] ?? seed()
        const next = { ...prev, sortBy, touchedAt: Date.now() }
        return { byWork: evict({ ...s.byWork, [key]: next }) }
      }),
    }),
    {
      name:    'typoon.work.list.ui.v1',
      storage: createJSONStorage(() => localStorage),
      // Only the byWork dictionary is worth persisting; the action
      // closures are recreated on every page load anyway.
      partialize: (s) => ({ byWork: s.byWork }),
    },
  ),
)


// ── Selector hooks ────────────────────────────────────────────


/** Subscribe to the persisted entry for a single work. Returns null
 *  when the user hasn't touched this work's list — caller decides the
 *  default (e.g. "filter target_lang if reachable, else 'all'").
 *  Using a focused selector keeps re-renders to "this work changed";
 *  edits to OTHER work entries don't re-render this consumer. */
export function useWorkListUi(workId: number): WorkListUiState | null {
  return useChapterListUi((s) => s.byWork[String(workId)] ?? null)
}


/** Stable setter pair — identity stays constant across renders via
 *  `useShallow` so consumers can drop them in dependency arrays
 *  without churn. */
export function useChapterListUiActions(): Pick<Store, 'setActiveLang' | 'setSortBy'> {
  return useChapterListUi(
    useShallow((s) => ({
      setActiveLang: s.setActiveLang,
      setSortBy:     s.setSortBy,
    })),
  )
}
