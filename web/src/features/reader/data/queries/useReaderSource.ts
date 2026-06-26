// useReaderSource — subscribe to the cache for one ActiveSource.
//
// Returns a stable source proxy so React components never see a
// closed source during cache transitions (eviction, re-open).

import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react'

import { useReaderCache } from '../../cache/ReaderCacheProvider'
import { openSourceFromActive } from '../../hooks/openSourceFromActive'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'
import { StableSourceProxy } from '../../sources'
import type { CacheEntry } from '../../cache/types'
import { type ActiveSource, type ReaderSourceState } from '../types'


export function useReaderSource(
  active: ActiveSource,
  key:    string,
): ReaderSourceState {
  const cache = useReaderCache()
  const sourceFetch = useSourceFetch()

  const activeRef = useRef(active)
  activeRef.current = active
  const sourceFetchRef = useRef(sourceFetch)
  sourceFetchRef.current = sourceFetch

  // Stable proxy — survives cache transitions without React re-mounts.
  const proxyRef = useRef<StableSourceProxy | null>(null)

  useEffect(() => {
    if (activeRef.current.kind === 'none') return

    cache.retain(key)
    void cache.open(key, signal =>
      openSourceFromActive(activeRef.current, signal, sourceFetchRef.current),
    ).catch(() => { })

    return () => cache.release(key)
  }, [cache, key, active.kind])

  const entry = useSyncExternalStore<CacheEntry | undefined>(
    listener => cache.subscribe(key, listener),
    () => cache.getEntry(key),
    () => cache.getEntry(key),
  )

  // Swap the proxy's inner source when the cache entry updates.
  // Done in an effect (not useMemo) because it's a side effect.
  useEffect(() => {
    if (entry?.status === 'ready' && proxyRef.current) {
      proxyRef.current.swap(entry.source)
    }
  }, [entry])

  // Close the proxy on unmount.
  useEffect(() => {
    return () => { proxyRef.current?.close() }
  }, [])

  return useMemo<ReaderSourceState>(() => {
    if (active.kind === 'none') return { status: 'no-source' }
    if (!entry)                 return { status: 'loading' }
    if (entry.status === 'opening') return { status: 'loading' }
    if (entry.status === 'error')   return { status: 'error', error: entry.error }

    // Create proxy on first ready entry.
    if (!proxyRef.current) {
      proxyRef.current = new StableSourceProxy(entry.source)
    }
    return { status: 'ready', source: proxyRef.current, mode: proxyRef.current.mode }
  }, [entry, active])
}
