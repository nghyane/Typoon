import { useEffect, useRef } from 'react'
import { createSSE } from '../api/events'
import type { SSEEvent } from '../api/types'

export function useSSE(onEvent: (event: SSEEvent) => void) {
  const cbRef = useRef(onEvent)
  cbRef.current = onEvent

  useEffect(() => {
    return createSSE((e) => cbRef.current(e))
  }, [])
}
