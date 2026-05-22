// Library — promotion layer over `features/works`.
//
// A "library" entry is just a Work with `in_library=true`. There is no
// separate library store. The hooks here are the promotion + querying
// surface for that flag — everything else (create, attach source, edit
// title) lives in `features/works/queries.ts`.

import { useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

import { db, type Work, type LibraryStatus } from '@shared/db'
import { qk } from '@shared/api/keys'

export type { Work, LibraryStatus }


// ── Reads ───────────────────────────────────────────────────────────

export interface LibraryFilter {
  status?: LibraryStatus
}

/** All pinned works, newest first. Browse-only works are excluded.
 *
 *  We filter `in_library` in JS because IndexedDB does not natively
 *  index booleans across browsers — Dexie's `*` would still scan, and
 *  the library is small (hundreds of pinned works at most). The
 *  `updated_at` index gives us the ordering. */
export function useLibraryWorks(filter?: LibraryFilter) {
  const status = filter?.status
  return useQuery({
    queryKey: status ? qk.library.byStatus(status) : qk.library.all(),
    queryFn:  async () => {
      const all = await db().works
        .orderBy('updated_at')
        .reverse()
        .filter(w => w.in_library && !w.deleted)
        .toArray()
      return status ? all.filter(w => w.library_status === status) : all
    },
    staleTime: Infinity,
  })
}

/** Counts per library_status — drives the status tabs without
 *  re-querying for each filter. */
export function useLibraryStatusCounts() {
  const lib = useLibraryWorks()
  return useMemo(() => {
    const works = lib.data ?? []
    const counts: Record<LibraryStatus | 'all', number> = {
      all: works.length, reading: 0, plan: 0, done: 0, dropped: 0,
    }
    for (const w of works) {
      if (w.library_status) counts[w.library_status] += 1
    }
    return counts
  }, [lib.data])
}


// ── Mutations ───────────────────────────────────────────────────────

/** Pin a Work into the library. Default status `'reading'`. Idempotent. */
export function useAddToLibrary() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      work_id: string
      status?: LibraryStatus
    }): Promise<Work> => {
      const cur = await db().works.get(args.work_id)
      if (!cur) throw new Error('Work không tồn tại.')
      const now = new Date().toISOString()
      const next: Work = {
        ...cur,
        in_library:       true,
        library_status:   args.status ?? cur.library_status ?? 'reading',
        library_added_at: cur.library_added_at ?? now,
        updated_at:       now,
      }
      await db().works.put(next)
      return next
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
    },
  })
}

/** Unpin a Work from the library. The Work itself stays (eligible for
 *  LRU prune), so the user can re-open it from history without losing
 *  reading position. */
export function useRemoveFromLibrary() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (workId: string): Promise<void> => {
      const cur = await db().works.get(workId)
      if (!cur) return
      const next: Work = {
        ...cur,
        in_library:     false,
        library_status: null,
        updated_at:     new Date().toISOString(),
      }
      await db().works.put(next)
    },
    onSuccess: (_v, workId) => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      qc.invalidateQueries({ queryKey: qk.works.byId(workId) })
    },
  })
}

export function useUpdateLibraryStatus() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: { work_id: string; status: LibraryStatus }): Promise<Work> => {
      const cur = await db().works.get(args.work_id)
      if (!cur) throw new Error('Work không tồn tại.')
      const next: Work = {
        ...cur,
        library_status: args.status,
        updated_at:     new Date().toISOString(),
      }
      await db().works.put(next)
      return next
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
    },
  })
}
