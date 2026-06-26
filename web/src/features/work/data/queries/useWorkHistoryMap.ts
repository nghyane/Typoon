// Bulk history live query — chapter_ref → HistoryItem for one Work.
//
// Wraps the existing `useWorkHistory` array query into a Map shape so
// chapter rows can do O(1) lookup for "is this chapter read".

import { useMemo } from 'react'

import { useWorkHistory as useWorkHistoryArray } from '@features/library/history'
import type { HistoryItem } from '../types'


const EMPTY: ReadonlyMap<string, HistoryItem> = new Map()


export function useWorkHistoryMap(
  workId: string | null | undefined,
): ReadonlyMap<string, HistoryItem> {
  const q = useWorkHistoryArray(workId)
  return useMemo(() => {
    if (!q.data) return EMPTY
    const m = new Map<string, HistoryItem>()
    for (const h of q.data) m.set(h.chapter_ref, h)
    return m
  }, [q.data])
}
