// useChapterPages — load all page blobs for a chapter.
//
// Accepts a stable `key` that changes only when the URL list actually
// changes.  The hook fetches whenever the key changes, using a reducer
// for clean state transitions — no ref tricks, no dependency hacks.

import { useEffect, useReducer } from 'react'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'

export interface PageBlobs {
  blobs: (Blob | null)[]
  done: number
  total: number
}

type Action =
  | { type: 'start'; total: number }
  | { type: 'loaded'; index: number; blob: Blob }
  | { type: 'error'; index: number }

function reducer(state: PageBlobs, action: Action): PageBlobs {
  switch (action.type) {
    case 'start': {
      if (state.total === action.total && state.blobs.some(b => b !== null)) return state
      return { blobs: new Array(action.total).fill(null), done: 0, total: action.total }
    }
    case 'loaded': {
      if (state.blobs[action.index] !== undefined && state.blobs[action.index] === null) {
        const next = [...state.blobs]
        next[action.index] = action.blob
        return { blobs: next, done: state.done + 1, total: state.total }
      }
      return state
    }
    case 'error': {
      const next = [...state.blobs]
      next[action.index] = null
      return { blobs: next, done: state.done + 1, total: state.total }
    }
  }
}

export function useChapterPages(
  rawUrls: readonly string[],
  key: string,
): PageBlobs {
  const sourceFetch = useSourceFetch()
  const [state, dispatch] = useReducer(reducer, {
    blobs: [], done: 0, total: rawUrls.length,
  })

  useEffect(() => {
    if (!rawUrls.length) return

    const ac = new AbortController()
    const total = rawUrls.length
    dispatch({ type: 'start', total })

    const CONCURRENCY = 6
    let next = 0

    const fetchOne = async (i: number): Promise<void> => {
      try {
        const proxied = sourceFetch.toBrowserUrl(rawUrls[i]!)
        const res = await fetch(proxied, { signal: ac.signal })
        if (!res.ok) throw new Error(`${res.status}`)
        const blob = await res.blob()
        if (!ac.signal.aborted) dispatch({ type: 'loaded', index: i, blob })
      } catch {
        if (!ac.signal.aborted) dispatch({ type: 'error', index: i })
      }
    }

    const worker = async (): Promise<void> => {
      while (!ac.signal.aborted) {
        const i = next++
        if (i >= total) break
        await fetchOne(i)
      }
    }

    Promise.allSettled(
      Array.from({ length: Math.min(CONCURRENCY, total) }, worker),
    )

    return () => ac.abort()
  }, [key, sourceFetch])

  return state
}
