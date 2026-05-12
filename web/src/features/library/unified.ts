// Unified library view-model — server library entries adapted for the
// /library page grid.
//
// Source of truth is `/api/library` (entries with `status` enum). This
// module flattens each entry into a `LibraryItem` carrying the
// status, target_lang, and a "hasNew" hint derived from the local
// reading-history store. The local store remains the place where
// per-source "Tiếp tục đọc" snapshots live; it powers the hint here
// but never overrides server state.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type ApiLibraryEntry, type LibraryStatus } from '@shared/api/api'
import { coverUrl } from '@shared/ui/Cover'
import { useLibrary, hasNewChapter } from './store'
import { useShallow } from 'zustand/react/shallow'

/** Filter chip identities for the /library page. `all` excludes
 *  `dropped` (the server applies the same filter when status param
 *  is omitted). */
export type LibraryFilter = 'all' | LibraryStatus

export interface LibraryItem {
  key:        string
  entryId:    number
  materialId: number | null

  title:    string
  cover:    string | null

  /** Server enums + flags driving visual treatment. */
  status:         LibraryStatus
  targetLang:     string | null
  autoTranslate:  boolean

  /** Local-derived: latest chapter at source differs from last-read. */
  hasNew:    boolean

  /** Most-recent meaningful timestamp for sort. */
  activity:     number
  /** Chapter label overlay on cover; from local reading-history. */
  chapterLabel: string | null
}

// ── adapter ───────────────────────────────────────────────────────

function fromEntry(
  e:               ApiLibraryEntry,
  hasNewFromLocal: boolean,
  chapterLabel:    string | null,
): LibraryItem {
  const last    = e.last_read_at ? Date.parse(e.last_read_at) : 0
  const created = e.created_at   ? Date.parse(e.created_at)   : 0
  return {
    key:           `entry::${e.id}`,
    entryId:       e.id,
    materialId:    e.primary_material_id,
    title:         e.title,
    cover:         coverUrl(e.cover_url, e.updated_at),
    status:        e.status,
    targetLang:    e.target_lang,
    autoTranslate: e.auto_translate,
    hasNew:        hasNewFromLocal,
    activity:      Math.max(last, created),
    chapterLabel,
  }
}

// ── sort ──────────────────────────────────────────────────────────

function compare(a: LibraryItem, b: LibraryItem): number {
  if (a.hasNew !== b.hasNew) return a.hasNew ? -1 : 1
  return b.activity - a.activity
}

/** Unified library data for the /library page.
 *
 *  When `filter='all'`, the server query omits the status param so
 *  the response naturally excludes `dropped`. Other filters narrow
 *  to a single status enum value.
 *
 *  Local reading-history supplies the "hasNew" badge + chapter-label
 *  overlay — both are derived state that doesn't need a round-trip. */
export function useUnifiedLibrary(filter: LibraryFilter): {
  items:   LibraryItem[]
  loading: boolean
  counts:  Record<LibraryFilter, number>
} {
  // We always fetch the unfiltered list so chip counts stay accurate
  // when the user toggles between filters. Server returns sans
  // `dropped` already; counts are computed client-side.
  const { data: entries = [], isPending } = useQuery({
    queryKey:  ['library'],
    queryFn:   () => api.listLibrary(),
    staleTime: 30_000,
  })

  const rawLocal = useLibrary(useShallow((s) => Object.values(s.items)))

  return useMemo(() => {
    // material_id → local row index. The local store is keyed by
    // (source, mangaUrl); we cross-reference via the optional
    // `materialId` field the import-material flow stamps on.
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

    // Counts pivot on status. `all` = items minus dropped (server
    // already filtered, so length equals all).
    const counts: Record<LibraryFilter, number> = {
      all: items.length,
      reading: 0, plan: 0, on_hold: 0, done: 0, dropped: 0,
    }
    for (const it of items) counts[it.status]++

    const filtered = filter === 'all'
      ? items
      : items.filter((it) => it.status === filter)

    return {
      items: filtered.sort(compare),
      loading: isPending,
      counts,
    }
  }, [entries, rawLocal, filter, isPending])
}
