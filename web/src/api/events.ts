import { sseUrl } from './client'
import type { SSEEvent } from './types'

export function createSSE(
  onEvent: (event: SSEEvent) => void,
  lastId = '0',
): () => void {
  let es: EventSource
  let closed = false

  function connect() {
    if (closed) return
    es = new EventSource(sseUrl('/api/events'))

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as SSEEvent
        lastId = e.lastEventId || lastId
        onEvent(data)
      } catch {}
    }

    es.onerror = () => {
      es.close()
      if (!closed) setTimeout(() => connect(), 3000)
    }
  }

  connect()
  return () => {
    closed = true
    es?.close()
  }
}
