import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from './api'
import { getToken } from './auth'

interface ServerEvent {
  type:        string
  chapter_id?: number
  project_id?: number
  stage?:      string
  page_index?: number
  page_total?: number
}

// Coarse-grained list invalidations are only triggered by lifecycle events,
// not by per-page progress (`PageDone` fires once per page). Without this
// filter a 200-page chapter would retrigger refetches on every page.
const PROJECT_LIST_TRIGGERS = new Set([
  'ChapterDownloaded',
  'ChapterDone',
  'ChapterFailed',
  'ChapterSkipped',
  'StageStarted',
  'StageDone',
  'StageFailed',
])

export function useServerEvents() {
  const qc = useQueryClient()

  useEffect(() => {
    const token = getToken()
    if (!token) return

    // EventSource has no header API; engine accepts the token via ?token=.
    const url = `${api.base}/api/events?token=${encodeURIComponent(token)}`
    const es  = new EventSource(url)

    es.onmessage = (msg) => {
      let ev: ServerEvent
      try { ev = JSON.parse(msg.data) }
      catch { return }

      if (ev.project_id) {
        qc.invalidateQueries({
          queryKey: ['projects', ev.project_id, 'chapters'],
        })
      }

      if (PROJECT_LIST_TRIGGERS.has(ev.type)) {
        qc.invalidateQueries({ queryKey: ['projects'], exact: true })
        if (ev.project_id) {
          qc.invalidateQueries({
            queryKey: ['projects', ev.project_id],
            exact: true,
          })
        }
      }
    }

    return () => es.close()
  }, [qc])
}
