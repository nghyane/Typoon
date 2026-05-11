// Library selectors — every derived list goes through `useShallow` so
// zustand doesn't re-trigger renders when the underlying array
// references change but contents don't (see docs/wiki/browse-mode.md §5).
//
// One unified `/library` page renders all entries from the same list,
// filtered by chip (not by route). "Chương mới" floats to top via the
// default sort, not via a separate tab — state belongs in visual
// treatment + sort, not in navigation. See chat history for the design
// rationale ("state should be visible, not hidden behind navigation").

import { useMemo } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { useLibrary, hasNewChapter, type LibraryEntry } from './store'

export type LibraryFilter = 'all' | 'reading' | 'bookmarks' | 'updates'

/** Most recent meaningful timestamp for an entry — used as fallback
 *  sort key when neither side is bookmarked nor read. */
function activityTime(e: LibraryEntry): number {
  return Math.max(e.lastReadAt ?? 0, e.bookmarkedAt ?? 0)
}

/** Default sort: hasNew first (so updates float to top of every
 *  filter view), then most-recent-activity. Stable across re-renders;
 *  unbookmarked-unread items can still appear (e.g. cards user opened
 *  once but never finished). */
function defaultCompare(a: LibraryEntry, b: LibraryEntry): number {
  const aNew = hasNewChapter(a) ? 1 : 0
  const bNew = hasNewChapter(b) ? 1 : 0
  if (aNew !== bNew) return bNew - aNew
  return activityTime(b) - activityTime(a)
}

function matchesFilter(e: LibraryEntry, filter: LibraryFilter): boolean {
  if (filter === 'all')       return true
  if (filter === 'reading')   return e.lastReadAt !== null
  if (filter === 'bookmarks') return e.bookmarked
  // updates
  return e.bookmarked && hasNewChapter(e)
}

export function useLibraryEntries(filter: LibraryFilter): LibraryEntry[] {
  return useLibrary(
    useShallow((s) =>
      Object.values(s.items)
        .filter((e) => matchesFilter(e, filter))
        .sort(defaultCompare),
    ),
  )
}

/** Source-scoped recent reads — used by /browse/$source landing's
 *  "Tiếp tục đọc" rail. Returns up to `limit` entries with non-null
 *  `lastReadAt`, newest first. */
export function useSourceContinueRail(sourceId: string, limit = 8): LibraryEntry[] {
  return useLibrary(
    useShallow((s) =>
      Object.values(s.items)
        .filter((e) => e.source === sourceId && e.lastReadAt !== null)
        .sort((a, b) => (b.lastReadAt ?? 0) - (a.lastReadAt ?? 0))
        .slice(0, limit),
    ),
  )
}

/** Counts for each filter chip. Single store pass. */
export function useLibraryCounts(): Record<LibraryFilter, number> {
  const flat = useLibrary(
    useShallow((s) => {
      let all = 0, reading = 0, bookmarks = 0, updates = 0
      for (const e of Object.values(s.items)) {
        all++
        if (e.lastReadAt !== null) reading++
        if (e.bookmarked) {
          bookmarks++
          if (hasNewChapter(e)) updates++
        }
      }
      return [all, reading, bookmarks, updates] as const
    }),
  )
  return useMemo(
    () => ({ all: flat[0], reading: flat[1], bookmarks: flat[2], updates: flat[3] }),
    [flat],
  )
}

/** One entry by (source, mangaUrl). Stable reference until that row
 *  actually changes — safe to use as a hook dep. */
export function useLibraryEntry(
  source: string | null | undefined,
  mangaUrl: string | null | undefined,
): LibraryEntry | null {
  return useLibrary((s) => {
    if (!source || !mangaUrl) return null
    return s.items[`${source}::${mangaUrl}`] ?? null
  })
}
