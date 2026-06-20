// openSourceFromActive — async factory used by the reader cache.
//
// Accepts an AbortSignal so the cache can cancel in-flight opens
// when an entry is evicted before resolving.

import {
  openRawOffline, openRawOnline,
  type ReaderSource,
} from '../sources'
import type { ActiveSource } from '../data/types'
import type { SourceFetch } from '@features/browse/proxy'


export async function openSourceFromActive(
  active: ActiveSource,
  signal: AbortSignal,
  sourceFetch: SourceFetch,
): Promise<ReaderSource> {
  if (signal.aborted) throw new DOMException('aborted', 'AbortError')

  switch (active.kind) {
    case 'none':
      throw new Error('no source')

    case 'raw-offline':
      return await openRawOffline(active.blob)

    case 'raw-online':
      return openRawOnline(active.urls, sourceFetch)
  }
}
