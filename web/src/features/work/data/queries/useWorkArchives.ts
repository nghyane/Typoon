// Bulk live query — all saved archives for one Work.
//
// Returns a Map keyed by `chapter_ref` so component consumers can do
// O(1) lookup per row instead of opening one IDB cursor per row.
//
// Live: re-runs on any write to `archives` (save / delete) thanks to
// `useLiveQuery`. The Map identity changes on every emit, so consumers
// using `useMemo` selectors derive cleanly.

import { useLiveQuery } from 'dexie-react-hooks'

import { db } from '@shared/db'
import type { SavedArchive } from '../types'


const EMPTY: ReadonlyMap<string, SavedArchive> = new Map()


export function useWorkArchives(
  workId: string | null | undefined,
): ReadonlyMap<string, SavedArchive> {
  const map = useLiveQuery(
    async () => {
      if (!workId) return EMPTY
      const rows = await db().archives.where('work_id').equals(workId).toArray()
      const m = new Map<string, SavedArchive>()
      for (const r of rows) m.set(r.chapter_ref, r)
      return m
    },
    [workId],
  )
  return map ?? EMPTY
}
