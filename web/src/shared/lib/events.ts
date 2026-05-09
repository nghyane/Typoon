import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { getToken } from '@features/auth/auth'

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

/**
 * Open one EventSource for the lifetime of a tab.
 *
 * `projectIds` narrows the server-side fan-out so a tab viewing a
 * single project doesn't receive PageDone events from every other
 * project that happens to be rendering. Pass undefined for firehose
 * (admin / queue dashboard).
 *
 * The hook reopens the stream when the project filter changes so the
 * server-side subscription matches what the user is actually viewing.
 */
export function useServerEvents(projectIds?: readonly number[]) {
  const qc = useQueryClient()
  const filterKey = projectIds?.length ? projectIds.slice().sort((a, b) => a - b).join(',') : ''

  useEffect(() => {
    const token = getToken()
    if (!token) return

    // EventSource has no header API; engine accepts the token via ?token=.
    const params = new URLSearchParams({ token })
    if (filterKey) params.set('projects', filterKey)
    const url = `${api.base}/api/events?${params}`
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
  }, [qc, filterKey])
}
