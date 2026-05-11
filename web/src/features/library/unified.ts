// Unified library item — view-model that merges entries from the
// local library store (external sources: HappyMH, MangaDex, …) with
// internal projects (community + mine + pinned) fetched from the
// typoon API. One card type, one grid, no cross-source branching at
// the view layer.
//
// Why this shape and not a generic `Manga` type:
//   • The 2 backends already exist (zustand + REST); changing either
//     forces a bigger refactor. Adapter pattern keeps phase A cheap.
//   • Card click target is the only place that branches. Everything
//     else (cover, title, hasNew tag, bookmark icon) is uniform.
//
// Phase B will collapse these into a single domain type.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type ApiProject } from '@shared/api/api'
import { coverUrl } from '@shared/ui/Cover'
import { useLibrary, hasNewChapter, type LibraryEntry } from './store'
import { useShallow } from 'zustand/react/shallow'
import type { LibraryFilter } from './hooks'

export interface LibraryItem {
  key:        string
  kind:       'external' | 'internal'
  source:     string             // manifest.id ("happymh" | "mangadex" | "community" | …)
  /** Identifier for routing: mangaUrl for external, project_id (as string) for internal. */
  ref:        string
  title:      string
  cover:      string | null
  /** Subtitle line under the title — source name for external,
   *  description excerpt for internal. */
  subtitle:   string | null

  /** Flags driving visual treatment + filter eligibility. */
  bookmarked:  boolean
  reading:     boolean             // has any read history
  hasNew:      boolean
  /** Internal-only — owner / shared status for badge. */
  ownership:   'mine' | 'shared' | 'pinned' | null

  /** Timestamps for sort. `activity` = most recent meaningful event. */
  activity:    number              // ms epoch
  /** For chapter label overlay on cover. Either last-read (external),
   *  or null for internal projects (no client-side chapter tracking). */
  chapterLabel: string | null
}

// ── adapters ──────────────────────────────────────────────────────

function fromExternal(e: LibraryEntry): LibraryItem {
  return {
    key:         `ext::${e.source}::${e.mangaUrl}`,
    kind:        'external',
    source:      e.source,
    ref:         e.mangaUrl,
    title:       e.title,
    cover:       e.cover,
    subtitle:    null,
    bookmarked:  e.bookmarked,
    reading:     e.lastReadAt !== null,
    hasNew:      hasNewChapter(e),
    ownership:   null,
    activity:    Math.max(e.lastReadAt ?? 0, e.bookmarkedAt ?? 0),
    chapterLabel: e.lastChapterRead?.label
      ?? e.latestChapter?.label
      ?? null,
  }
}

function fromInternal(p: ApiProject): LibraryItem {
  const ownership: LibraryItem['ownership'] =
    p.is_owner ? 'mine'
    : p.is_pinned ? 'pinned'
    : 'shared'
  const activity = p.updated_at ? Date.parse(p.updated_at) : 0
  return {
    key:         `int::${p.project_id}`,
    kind:        'internal',
    source:      'community',
    ref:         String(p.project_id),
    title:       p.title,
    cover:       coverUrl(p.cover_url, p.updated_at),
    subtitle:    p.description?.trim() || null,
    // Internal `is_pinned` maps to the same "bookmarked" UX — the
    // user sees one consistent bookmark icon across all entries.
    bookmarked:  p.is_pinned,
    reading:     false,            // no per-project history in the API yet
    hasNew:      false,            // server can light this up later
    ownership,
    activity,
    chapterLabel: null,
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

/** Single hook that returns the merged + filtered + sorted list.
 *  Internal projects load via React Query (background revalidate);
 *  external entries are local-first. Empty result while internal
 *  query is pending — render falls back to external-only meanwhile. */
export function useUnifiedLibrary(filter: LibraryFilter): {
  items:     LibraryItem[]
  loading:   boolean
  counts:    Record<LibraryFilter, number>
} {
  // IMPORTANT: select raw entries (whose object identity is stable
  // unless the store actually mutates them), then map outside the
  // selector. Calling `.map(fromExternal)` inside the selector would
  // produce fresh object literals every render → useShallow's
  // element-wise compare returns false → re-render loop. See
  // docs/wiki/browse-mode.md §5 ("Maximum update depth exceeded").
  const rawExternal = useLibrary(
    useShallow((s) => Object.values(s.items)),
  )

  // `filter=all` gives us mine + shared + pinned in one request.
  const { data: projects = [], isPending } = useQuery({
    queryKey: ['projects', 'all'],
    queryFn:  () => api.listProjects('all'),
    staleTime: 60_000,
  })

  return useMemo(() => {
    const external = rawExternal.map(fromExternal)
    const internal = projects.map(fromInternal)
    const merged   = [...external, ...internal]

    let all = 0, reading = 0, bookmarks = 0, updates = 0
    for (const it of merged) {
      all++
      if (it.reading) reading++
      if (it.bookmarked) {
        bookmarks++
        if (it.hasNew) updates++
      }
    }

    const items = merged.filter((it) => matches(it, filter)).sort(compare)
    return {
      items,
      loading: isPending,
      counts:  { all, reading, bookmarks, updates },
    }
  }, [rawExternal, projects, filter, isPending])
}
