// Unified library item — view-model adapter on top of /api/library
// (server) and the local reading-history store (browser-only).
//
// Backend owns: bookmark flag, primary_material_id, last_read_at,
//               linked materials.
// Local owns:   per-source "Tiếp tục đọc" snapshots (latest chapter
//               we saw on each material so the chapter-row "Mới"
//               badge can fire without a backend round-trip).
//
// The two are joined by material_id: a library entry's primary
// material_id matches a local store key (`${source}::${mangaUrl}`)
// via the per-material lookup the route does at /api/material/import.
// In this slice we lean entirely on the server entry for the grid;
// local history augments per-source rails (BrowseSourceHome) only.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type ApiLibraryEntry } from '@shared/api/api'
import { coverUrl } from '@shared/ui/Cover'
import { useLibrary, hasNewChapter } from './store'
import { useShallow } from 'zustand/react/shallow'
import type { LibraryFilter } from './hooks'

export interface LibraryItem {
  key:      string
  /** Server library_entry id. Route uses this for /library/entry/$id
   *  bookmark + unlink actions. */
  entryId:  number
  /** Primary material the entry routes to (the "default version" the
   *  user clicked through last). */
  materialId: number | null

  title:    string
  cover:    string | null

  /** Flags driving visual treatment + filter eligibility. */
  bookmarked: boolean
  reading:    boolean
  hasNew:     boolean

  /** Timestamps for sort. `activity` = most recent meaningful event. */
  activity:    number
  /** Chapter label overlay on cover; populated from the local
   *  reading-history store when we have a recent snapshot. */
  chapterLabel: string | null
}

// ── adapter ───────────────────────────────────────────────────────

function fromEntry(
  e: ApiLibraryEntry,
  hasNewFromLocal: boolean,
  chapterLabel:    string | null,
): LibraryItem {
  const last = e.last_read_at ? Date.parse(e.last_read_at) : 0
  const book = e.bookmarked_at ? Date.parse(e.bookmarked_at) : 0
  return {
    key:        `entry::${e.id}`,
    entryId:    e.id,
    materialId: e.primary_material_id,
    title:      e.title,
    cover:      coverUrl(e.cover_url, e.updated_at),
    bookmarked: e.bookmarked,
    reading:    e.last_read_at !== null,
    hasNew:     hasNewFromLocal,
    activity:   Math.max(last, book),
    chapterLabel,
  }
}

// ── merge + sort ──────────────────────────────────────────────────

function compare(a: LibraryItem, b: LibraryItem): number {
  if (a.hasNew !== b.hasNew) return a.hasNew ? -1 : 1
  return b.activity - a.activity
}

function matches(it: LibraryItem, filter: LibraryFilter): boolean {
  if (filter === 'all')       return true
  if (filter === 'reading')   return it.reading
  if (filter === 'bookmarks') return it.bookmarked
  return it.bookmarked && it.hasNew     // updates
}

/** Single hook returning filtered + sorted library items.
 *
 *  Source of truth is `/api/library`. Local reading-history store
 *  supplies the per-entry "hasNew" flag and chapter-label overlay —
 *  both are reader-side derived state that doesn't need a round-trip.
 */
export function useUnifiedLibrary(filter: LibraryFilter): {
  items:   LibraryItem[]
  loading: boolean
  counts:  Record<LibraryFilter, number>
} {
  const { data: entries = [], isPending } = useQuery({
    queryKey: ['library'],
    queryFn:  () => api.listLibrary(),
    staleTime: 30_000,
  })

  // Pull the local reading-history rows once and turn them into a
  // material-id-keyed lookup. Local entries are keyed by source+url
  // but the server entry only knows `primary_material_id`; the
  // material-import route returns the material row which is how we'll
  // bridge the two going forward. For now we expose history by
  // material_id when the local store has annotated one (slice
  // future-stamp: local store will track material_id alongside).
  const rawLocal = useLibrary(useShallow((s) => Object.values(s.items)))

  return useMemo(() => {
    // Build a quick lookup for material_id → local row when present
    // (the field is added in a follow-up slice; defensive read).
    const byMaterial = new Map<number, typeof rawLocal[number]>()
    for (const e of rawLocal) {
      const mid = (e as unknown as { materialId?: number }).materialId
      if (mid != null) byMaterial.set(mid, e)
    }

    const items: LibraryItem[] = entries.map((e) => {
      const local = e.primary_material_id != null
        ? byMaterial.get(e.primary_material_id)
        : undefined
      return fromEntry(
        e,
        local ? hasNewChapter(local) : false,
        local?.lastChapterRead?.label ?? local?.latestChapter?.label ?? null,
      )
    })

    let all = 0, reading = 0, bookmarks = 0, updates = 0
    for (const it of items) {
      all++
      if (it.reading) reading++
      if (it.bookmarked) {
        bookmarks++
        if (it.hasNew) updates++
      }
    }

    const filtered = items.filter((it) => matches(it, filter)).sort(compare)
    return {
      items: filtered,
      loading: isPending,
      counts:  { all, reading, bookmarks, updates },
    }
  }, [entries, rawLocal, filter, isPending])
}
