// Library hooks — derive view-model lists from local per-source
// reading history (`./store`). The Library page chip filter operates
// on the SERVER `library_entry.status` enum, not this store; this
// module powers per-source "Tiếp tục đọc" rails only.
//
// Server status enum (reading/plan/on_hold/done/dropped) is the
// source of truth for the unified Library page — see `./unified.ts`.

import { useShallow } from 'zustand/react/shallow'
import { useLibrary, type LibraryEntry } from './store'

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
