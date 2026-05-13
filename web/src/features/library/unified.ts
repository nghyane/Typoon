// Unified library view-model — server library entries adapted for
// the /library grid.
//
// One grid. Filter chips combine reading status (5 enum values) with
// translation activity (translating / errored). Status filters live
// on `library_entry.status`; activity filters pivot on
// `translation_summary`. Both kinds funnel through the same chip row
// — manga and "bản dịch" are not separate views, just different
// slices of the same list.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api, type ApiLibraryEntry, type ApiTranslationSummary, type LibraryStatus,
} from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { coverUrl } from '@shared/ui/Cover'

/** Filter chip identities. `all` shows every visible entry. Activity
 *  filters (`translating`, `errored`) cut across status — they're
 *  answered by translation_summary not the enum.
 *
 *  `dropped` isn't a UI status anymore: the StatusPicker only offers
 *  reading / plan / done plus a destructive "Bỏ theo dõi" that deletes
 *  the entry. Legacy entries with status='dropped' still load but
 *  filter under `all` only — no dedicated chip. */
export type LibraryFilter =
  | 'all'
  | 'reading' | 'plan' | 'done'
  | 'translating'
  | 'errored'

export interface LibraryItem {
  key:        string
  entryId:    number
  workId:     number

  title:    string
  cover:    string | null

  status:    LibraryStatus

  /** Activity summary (only this user's translations). */
  summary:    ApiTranslationSummary

  /** Local-derived: latest chapter at source differs from last-read. */
  hasNew:       boolean

  /** Most-recent meaningful timestamp for sort. */
  activity:     number
  /** Chapter label overlay on cover; from local reading-history. */
  chapterLabel: string | null
}

function fromEntry(
  e:               ApiLibraryEntry,
  hasNewFromLocal: boolean,
  chapterLabel:    string | null,
): LibraryItem {
  const updated = e.updated_at ? Date.parse(e.updated_at) : 0
  const created = e.created_at ? Date.parse(e.created_at) : 0
  const summary: ApiTranslationSummary = e.translation_summary ?? {
    pending: 0, running: 0, done: 0, error: 0,
  }
  return {
    key:           `entry::${e.id}`,
    entryId:       e.id,
    workId:        e.work_id,
    title:         e.title,
    cover:         coverUrl(e.cover_url, e.updated_at),
    status:        e.status,
    summary,
    hasNew:        hasNewFromLocal,
    activity:      Math.max(updated, created),
    chapterLabel,
  }
}

function compare(a: LibraryItem, b: LibraryItem): number {
  // Running translations float to the top — they're what the user
  // actively wants to check on.
  const aRun = a.summary.running > 0 ? 1 : 0
  const bRun = b.summary.running > 0 ? 1 : 0
  if (aRun !== bRun) return bRun - aRun
  if (a.hasNew !== b.hasNew) return a.hasNew ? -1 : 1
  return b.activity - a.activity
}

function matches(it: LibraryItem, filter: LibraryFilter): boolean {
  if (filter === 'all')         return true
  if (filter === 'translating') return it.summary.running > 0 || it.summary.pending > 0
  if (filter === 'errored')     return it.summary.error > 0
  return it.status === filter
}

/** Unified library data. Always fetches the unfiltered list so chip
 *  counts stay accurate; client-side filter pivots on the union of
 *  status enum + activity buckets. */
export function useUnifiedLibrary(filter: LibraryFilter): {
  items:   LibraryItem[]
  loading: boolean
  counts:  Record<LibraryFilter, number>
} {
  const { data: entries = [], isPending } = useQuery({
    queryKey:  qk.library.all(),
    queryFn:   () => api.listLibrary(),
    staleTime: 30_000,
  })

  return useMemo(() => {
    // Local reading-state lives under (source, mangaUrl) which doesn't
    // map cleanly to a Work id; until we surface "last read" via the
    // server reading_history, the local-derived `hasNew` falls back
    // to false. Cover-overlay chapter label likewise empty.
    const items: LibraryItem[] = entries.map((e) =>
      fromEntry(e, false, null),
    )

    const counts: Record<LibraryFilter, number> = {
      all: items.length,
      reading: 0, plan: 0, done: 0,
      translating: 0, errored: 0,
    }
    for (const it of items) {
      // Legacy 'dropped' entries don't have a chip; they still
      // count toward `all` so the user can find them.
      if (it.status === 'reading' || it.status === 'plan' || it.status === 'done') {
        counts[it.status]++
      }
      if (it.summary.running > 0 || it.summary.pending > 0) counts.translating++
      if (it.summary.error   > 0)                            counts.errored++
    }

    const filtered = items.filter((it) => matches(it, filter))

    return {
      items: filtered.sort(compare),
      loading: isPending,
      counts,
    }
  }, [entries, filter, isPending])
}
