// Resolve "translated archive" for a (work_id, chapter_ref).
//
// Looks in IndexedDB first (mirrored from /me/jobs). If a matching JobRef
// exists with state='done' and a fresh archive_url, return it. Otherwise
// return null and let the caller render the raw chapter + a "Dịch chương
// này" button.
//
// The presigned archive URL has a short TTL (1h). When expired, the
// caller should call `useRefreshArchiveUrl(jobId)` which hits
// `GET /jobs/:id` and updates IDB with a fresh URL.

import { useCallback } from 'react'
import { useLiveQuery } from 'dexie-react-hooks'
import { db, type JobRef } from '@shared/db'
import { api } from '@shared/api/api'


export interface ArchiveLookupResult {
  job_ref:      JobRef | null
  archive_url:  string | null
  /** True when we have a JobRef but the URL has expired. Caller should
   *  trigger a refresh via /api/jobs/:id. */
  needs_refresh: boolean
}

export function useResolveArchive(
  work_id:     string | null | undefined,
  chapter_ref: string | null | undefined,
): ArchiveLookupResult {
  const ref = useLiveQuery(async () => {
    if (!work_id || !chapter_ref) return null
    const rows = await db().jobs
      .where({ work_id, chapter_ref })
      .filter(j => j.kind === 'translate' && j.state === 'done')
      .reverse()
      .sortBy('created_at')
    return rows[0] ?? null
  }, [work_id, chapter_ref])

  if (!ref) return { job_ref: null, archive_url: null, needs_refresh: false }

  const expired = ref.archive_expires
    ? Date.parse(ref.archive_expires) < Date.now()
    : true
  if (expired || !ref.archive_url) {
    return { job_ref: ref, archive_url: null, needs_refresh: true }
  }
  return { job_ref: ref, archive_url: ref.archive_url, needs_refresh: false }
}

/** Hits /api/jobs/:id and updates IndexedDB with a fresh archive_url. */
export function useRefreshArchiveUrl() {
  return useCallback(async (id: number): Promise<string | null> => {
    const job = await api.jobsGet(id)
    if (job.state !== 'done' || !job.archive_url) return null
    await db().jobs.update(id, {
      archive_url:     job.archive_url,
      archive_expires: new Date(Date.now() + 3_600_000).toISOString(),
    })
    return job.archive_url
  }, [])
}
