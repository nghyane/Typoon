// Jobs feature — server job lifecycle + IndexedDB mirror.
//
//   useJob(id)             live status (polled every 2 s while running)
//   useMyJobs()             list of last 7 d jobs (server) + IDB mirror
//   useDeleteJob()          mutation, invalidates list + clears IDB row
//   useUserEventsStream()   single WS connection per device — receives a
//                           snapshot on connect + per-job pushes thereafter.
//                           Drives IDB mirror + TanStack cache for every
//                           consumer in the app (work page, jobs list,
//                           reader, etc.) without per-job sockets.
//
// Upload + start orchestration lives in `useTranslateChapter` /
// `useAnalyzeChapter` (next module) — this file is pure data layer.

import { useEffect } from 'react'
import {
  useQuery, useMutation, useQueryClient,
} from '@tanstack/react-query'

import { api, type ApiJob } from '@shared/api/api'
import { qk }   from '@shared/api/keys'
import { db, type JobRef } from '@shared/db'


// ── Wire → IDB conversion ────────────────────────────────────────────

function toJobRef(job: ApiJob): JobRef {
  return {
    id:              job.id,
    work_id:         job.work_id,
    chapter_ref:     null,    // upstream layer (translate hook) writes this
    kind:            job.kind,
    state:           job.state,
    archive_url:     job.archive_url,
    archive_expires: job.archive_url
      ? new Date(Date.now() + 3_600_000).toISOString()
      : null,
    page_count:      job.page_count,
    created_at:      job.created_at,
    expires_at:      job.expires_at,
  }
}

async function mirrorJob(job: ApiJob): Promise<void> {
  const existing = await db().jobs.get(job.id)
  const ref = toJobRef(job)
  // Preserve chapter_ref set by the translate hook on initial put.
  if (existing?.chapter_ref) ref.chapter_ref = existing.chapter_ref
  await db().jobs.put(ref)
}


// ── Queries ──────────────────────────────────────────────────────────

const POLL_STATES_MS: Record<ApiJob['state'], number | false> = {
  init:      2000,
  uploading: 2000,
  pending:   2000,
  running:   2000,
  done:      false,
  error:     false,
  expired:   false,
}

export function useJob(id: number | null | undefined) {
  return useQuery({
    queryKey: id ? qk.jobs.byId(id) : ['jobs', 'invalid'],
    queryFn:  async () => {
      const job = await api.jobsGet(id!)
      await mirrorJob(job)
      return job
    },
    enabled:                !!id,
    refetchInterval:        (q) => POLL_STATES_MS[(q.state.data as ApiJob | undefined)?.state ?? 'init'] ?? false,
    refetchIntervalInBackground: false,
  })
}

export function useMyJobs() {
  return useQuery({
    queryKey: qk.jobs.list(),
    queryFn:  async () => {
      const jobs = await api.jobsList()
      // Mirror all into IDB so the reader can lookup by work_id offline.
      await Promise.all(jobs.map(mirrorJob))
      return jobs
    },
    staleTime: 30_000,
  })
}


// ── Mutations ────────────────────────────────────────────────────────

export function useDeleteJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => api.jobsDelete(id),
    onSuccess:  async (_v, id) => {
      await db().jobs.delete(id)
      qc.invalidateQueries({ queryKey: qk.jobs.list() })
      qc.removeQueries({ queryKey: qk.jobs.byId(id) })
    },
  })
}


// ── Per-user WebSocket: one connection multiplexes every job ─────────

type UserEvent =
  | { kind: 'snapshot'; jobs: ApiJob[] }
  | { kind: 'job';      job:  ApiJob }

/** Mount once at the root of the authenticated app. The WS pushes a fresh
 *  `ApiJob` after every D1 mutation server-side, so IDB + TanStack cache
 *  stay reactive across every consumer (work page, jobs list, reader)
 *  without anyone polling per-job. Reconnects with exponential backoff on
 *  drop; the server re-sends a full snapshot on connect so any updates
 *  missed while disconnected are picked up. */
export function useUserEventsStream(enabled: boolean): void {
  const qc = useQueryClient()
  useEffect(() => {
    if (!enabled) return

    let ws:        WebSocket | null = null
    let cancelled = false
    let attempt   = 0
    let timer:    number | null = null

    const applyJob = async (job: ApiJob): Promise<void> => {
      await mirrorJob(job)
      qc.setQueryData(qk.jobs.byId(job.id), job)
      qc.setQueryData<ApiJob[]>(qk.jobs.list(), (prev) => {
        if (!prev) return prev
        const idx = prev.findIndex(j => j.id === job.id)
        if (idx === -1) return [job, ...prev]
        const next = prev.slice()
        next[idx] = job
        return next
      })
    }

    const connect = () => {
      if (cancelled) return
      ws = new WebSocket(api.meEventsWsUrl())
      ws.addEventListener('open', () => { attempt = 0 })
      ws.addEventListener('message', async (ev) => {
        try {
          const data = JSON.parse(ev.data) as UserEvent
          if (data.kind === 'snapshot') {
            await Promise.all(data.jobs.map(mirrorJob))
            qc.setQueryData(qk.jobs.list(), data.jobs)
            for (const j of data.jobs) qc.setQueryData(qk.jobs.byId(j.id), j)
          } else if (data.kind === 'job') {
            await applyJob(data.job)
          }
        } catch {
          /* malformed frame — ignore */
        }
      })
      ws.addEventListener('close', () => {
        if (cancelled) return
        attempt += 1
        const delay = Math.min(30_000, 1_000 * 2 ** Math.min(attempt, 5))
        timer = window.setTimeout(connect, delay)
      })
    }

    connect()

    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
      if (ws) { try { ws.close() } catch { /* */ } }
    }
  }, [enabled, qc])
}
