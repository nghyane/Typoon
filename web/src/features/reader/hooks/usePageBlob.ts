// usePageBlob — fetch + cache a single page as a Blob + ImageBitmap.
//
// No <img>, no blob URLs, no re-fetch.  The same blob feeds both
// canvas display and the translation pipeline.

import { useCallback, useEffect, useRef, useState } from 'react'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'
import type { ActiveSource } from '../../reader/data/types'

export type PageBlobStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface PageBlob {
  index: number
  blob: Blob | null
  bitmap: ImageBitmap | null
  status: PageBlobStatus
  error?: string
}

export function usePageBlob(
  active: ActiveSource,
  index: number,
): PageBlob {
  const sourceFetch = useSourceFetch()
  const [state, setState] = useState<PageBlob>({
    index, blob: null, bitmap: null, status: 'idle',
  })
  const fetchingRef = useRef(false)

  const fetchPage = useCallback(async () => {
    if (fetchingRef.current || state.status === 'ready') return
    fetchingRef.current = true
    setState(s => ({ ...s, status: 'loading' }))

    try {
      const url = resolveUrl(active, index)
      if (!url) throw new Error(`no URL for page ${index}`)

      const proxied = sourceFetch.toBrowserUrl(url)
      const res = await globalThis.fetch(proxied)
      if (!res.ok) throw new Error(`fetch page ${index}: ${res.status}`)

      const blob = await res.blob()
      const bitmap = await createImageBitmap(blob)

      setState({ index, blob, bitmap, status: 'ready' })
    } catch (err) {
      setState(s => ({ ...s, status: 'error', error: String(err) }))
    } finally {
      fetchingRef.current = false
    }
  }, [active, index, sourceFetch, state.status])

  useEffect(() => {
    if (state.status === 'idle') fetchPage()
  }, [state.status, fetchPage])

  // Cleanup bitmap on unmount.
  useEffect(() => {
    return () => { state.bitmap?.close() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { ...state, retry: () => { fetchingRef.current = false; setState(s => ({ ...s, status: 'idle' })) } } as PageBlob & { retry(): void }
}

function resolveUrl(active: ActiveSource, index: number): string | null {
  if (active.kind === 'none') return null
  if (active.kind === 'raw-online') return active.urls[index] ?? null
  if (active.kind === 'raw-offline') {
    // Bunle sources are handled differently — they have their own URL resolution.
    // For now, offline sources don't use this hook.
    return null
  }
  return null
}
