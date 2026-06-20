// usePageUrl — resolve a blob URL for one page of a source.
//
// State lives here (useState), not in a separate class. Preloading
// happens fire-and-forget in the view layer — no pub/sub needed.

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReaderSource } from '../sources'

export type PageStatus = 'idle' | 'loading' | 'ready' | 'error'

interface State {
  status:  PageStatus
  url:     string | null
  error:   Error | null
}

export function usePageUrl(source: ReaderSource, index: number) {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<State>({ status: 'idle', url: null, error: null })

  const mountedRef = useRef(source)
  mountedRef.current = source

  useEffect(() => {
    let cancelled = false
    setState({ status: 'loading', url: null, error: null })

    source.getUrl(index)
      .then(url => {
        if (!cancelled && mountedRef.current === source) {
          setState({ status: 'ready', url, error: null })
        }
      })
      .catch(error => {
        if (!cancelled && mountedRef.current === source) {
          setState({ status: 'error', url: null, error: error as Error })
        }
      })

    return () => { cancelled = true }
  }, [source, index, attempt])

  const retry = useCallback(() => {
    setAttempt(n => n + 1)
  }, [])

  return { status: state.status, url: state.url, error: state.error, retry }
}
