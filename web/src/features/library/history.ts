// Reading history — last-read page per Work × chapter.
//
// IndexedDB-backed, no server. The reader writes here on every page
// change (debounced). Continue-reading rails read from here joined with
// `works` to surface "still active" items.
//
// Chapter labels are denormalized so the UI can render "Đang đọc · Ch. 12"
// without round-tripping the source manifest.

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { db, type HistoryItem, type Work } from '@shared/db'
import { qk } from '@shared/api/keys'

export type { HistoryItem }


function historyId(work_id: string, chapter_ref: string): string {
  return `${work_id}:${chapter_ref}`
}


// ── Reads ───────────────────────────────────────────────────────────

export function useRecentReads(limit = 30) {
  return useQuery({
    queryKey: qk.history.all(),
    queryFn:  () =>
      db().history.orderBy('last_read_at').reverse().limit(limit).toArray(),
    staleTime: Infinity,
  })
}

export function useWorkHistory(workId: string | null | undefined) {
  return useQuery({
    queryKey: workId ? qk.history.forWork(workId) : ['history', 'invalid'],
    queryFn:  () =>
      db().history.where('work_id').equals(workId!).toArray(),
    enabled:   !!workId,
    staleTime: Infinity,
  })
}


// ── Continue reading ────────────────────────────────────────────────

export interface ContinueReadingRow {
  work:    Work
  history: HistoryItem
}

/** Most recently-read chapters joined with their parent Work. Used by
 *  the Home "Tiếp tục đọc" rail. Works that have been deleted are
 *  filtered out so the rail never shows orphans. */
export function useContinueReading(limit = 8) {
  return useQuery({
    queryKey: ['history', 'continue', limit],
    queryFn:  async () => {
      const recent = await db().history
        .orderBy('last_read_at')
        .reverse()
        .limit(limit * 2)        // overfetch in case some works are deleted
        .toArray()
      if (recent.length === 0) return []

      const ids = [...new Set(recent.map(h => h.work_id))]
      const works = await db().works.bulkGet(ids)
      const map = new Map<string, Work>()
      for (const w of works) {
        if (w && !w.deleted) map.set(w.id, w)
      }

      const rows: ContinueReadingRow[] = []
      for (const h of recent) {
        const w = map.get(h.work_id)
        if (w) rows.push({ work: w, history: h })
        if (rows.length >= limit) break
      }
      return rows
    },
    staleTime: 30_000,
  })
}


// ── Mutations ───────────────────────────────────────────────────────

export interface RecordReadingArgs {
  work_id:        string
  chapter_ref:    string
  chapter_label?: string
  page:           number
  total_pages?:   number | null
}

export function useRecordReading() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: RecordReadingArgs) => {
      const id = historyId(args.work_id, args.chapter_ref)
      const prev = await db().history.get(id)
      await db().history.put({
        id,
        work_id:       args.work_id,
        chapter_ref:   args.chapter_ref,
        chapter_label: args.chapter_label ?? prev?.chapter_label ?? args.chapter_ref,
        page:          args.page,
        total_pages:   args.total_pages ?? prev?.total_pages ?? null,
        last_read_at:  new Date().toISOString(),
      })
    },
    onSuccess: (_v, args) => {
      qc.invalidateQueries({ queryKey: qk.history.all() })
      qc.invalidateQueries({ queryKey: qk.history.forWork(args.work_id) })
      qc.invalidateQueries({ queryKey: ['history', 'continue'] })
    },
  })
}
