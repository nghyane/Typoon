import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { getToken } from '@features/auth/auth'

interface ProjectEvent {
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
 * Subscribe to live events for one project.
 *
 * The server-side stream is Postgres LISTEN scoped to a per-project
 * channel — a tab open on /projects/9 receives events about project 9
 * only, with no app-side filtering. Closing the tab drops the LISTEN.
 *
 * Pass the project id of the page the user is currently viewing. Any
 * non-positive integer (NaN, 0, undefined) disables the subscription
 * so callers can pass `Number(routeParam)` without re-validating.
 */
export function useProjectEvents(projectId: number | null | undefined) {
  const qc = useQueryClient()
  const id = Number.isInteger(projectId) && (projectId as number) > 0
    ? (projectId as number)
    : null

  useEffect(() => {
    if (id === null) return
    const token = getToken()
    if (!token) return

    // EventSource has no header API; engine accepts the token via ?token=.
    const url = `${api.base}/api/projects/${id}/events?token=${encodeURIComponent(token)}`
    const es  = new EventSource(url)

    es.onmessage = (msg) => {
      let ev: ProjectEvent
      try { ev = JSON.parse(msg.data) }
      catch { return }

      qc.invalidateQueries({
        queryKey: ['projects', id, 'chapters'],
      })

      if (PROJECT_LIST_TRIGGERS.has(ev.type)) {
        // Bubble up coarse changes so the global project list reflects
        // chapter completions / errors without each tab polling.
        qc.invalidateQueries({ queryKey: ['projects'], exact: true })
        qc.invalidateQueries({
          queryKey: ['projects', id],
          exact: true,
        })
      }
    }

    return () => es.close()
  }, [id, qc])
}