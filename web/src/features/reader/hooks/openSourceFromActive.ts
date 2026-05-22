// openSourceFromActive — async factory used by the reader cache.
//
// Accepts an AbortSignal so the cache can cancel in-flight opens
// when an entry is evicted before resolving.

import {
  openTranslatedOnline, openTranslatedOffline,
  openRawOffline, openRawOnline,
  type ReaderSource,
} from '../sources'
import { api } from '@shared/api/api'
import { db } from '@shared/db'
import type { ActiveSource } from '../data/types'


export async function openSourceFromActive(
  active: ActiveSource,
  signal: AbortSignal,
): Promise<ReaderSource> {
  if (signal.aborted) throw new DOMException('aborted', 'AbortError')

  switch (active.kind) {
    case 'none':
      throw new Error('no source')

    case 'translated-offline':
      return await openTranslatedOffline(active.blob)

    case 'translated-online': {
      let url = active.archiveUrl
      if (active.needsRefresh || !url) {
        const job = await api.jobsGet(active.jobId)
        if (signal.aborted) throw new DOMException('aborted', 'AbortError')
        if (job.state !== 'done' || !job.archive_url) {
          throw new Error('archive not ready')
        }
        url = job.archive_url
        await db().jobs.update(active.jobId, {
          archive_url:     url,
          archive_expires: new Date(Date.now() + 3_600_000).toISOString(),
        })
      }
      return await openTranslatedOnline(url)
    }

    case 'raw-offline':
      return await openRawOffline(active.blob)

    case 'raw-online':
      return openRawOnline(active.urls)
  }
}
