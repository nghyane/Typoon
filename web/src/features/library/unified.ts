// Unified library view-model — server library entries adapted for
// the /library grid + translation view.
//
// Source of truth is `/api/library`. The local reading-history store
// still owns per-source "Tiếp tục đọc" snapshots; this module joins
// the two by primary_material_id when the local store knows it.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api, type ApiLibraryEntry, type ApiTranslationSummary, type LibraryStatus,
} from '@shared/api/api'
import { coverUrl } from '@shared/ui/Cover'
import { useLibrary, hasNewChapter } from './store'
import { useShallow } from 'zustand/react/shallow'

/** Filter chip identities for the /library page. `all` excludes
 *  `dropped`. */
export type LibraryFilter = 'all' | LibraryStatus

export interface LibraryItem {
  key:        string
  entryId:    number
  materialId: number | null

  title:    string
  cover:    string | null

  status:         LibraryStatus
  targetLang:     string | null
  autoTranslate:  boolean

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
    summary:       e.translation_summary,
    hasNew:        hasNewFromLocal,
    activity:      Math.max(last, created),
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

/** Unified library data. Always fetches the unfiltered list so chip
 *  counts stay accurate; client-side filter pivots on `status`. */
export function useUnifiedLibrary(filter: LibraryFilter): {
  items:   LibraryItem[]
  loading: boolean
  counts:  Record<LibraryFilter, number>
} {
  const { data: entries = [], isPending } = useQuery({
    queryKey:  ['library'],
    queryFn:   () => api.listLibrary(),
    staleTime: 30_000,
  })

  const rawLocal = useLibrary(useShallow((s) => Object.values(s.items)))

  return useMemo(() => {
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
