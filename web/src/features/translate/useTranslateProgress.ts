/** Translate progress hook — subscribes to /api/translate/{id}/events.
 *
 *  SSE delivers `StageStarted` / `PageDone` / `StageDone` / `StageFailed`
 *  events for the draft + per-translation render. We collapse them into
 *  a single { stage, index, total, state, error } snapshot the UI binds
 *  to. The stream auto-closes on unmount and on terminal state.
 */

import { useEffect, useState } from 'react'
import { api } from '@shared/api/api'
import { getToken } from '@features/auth/auth'

export type ProgressState = 'pending' | 'running' | 'done' | 'error'

export interface TranslateProgress {
  state:        ProgressState
  /** Current stage: 'scan' | 'translate' | 'render' | '' */
  stage:        string
  /** Page or bubble index for the running stage. */
  index:        number
  /** Total units for the running stage. */
  total:        number
  /** Error message when state='error'. */
  error?:       string
  /** Public archive URL when done. Polled via /api/translate/{id}
   *  after render completes; we don't carry it on the SSE payload to
   *  keep events lean. */
  archiveUrl?:  string | null
}

interface RawEvent {
  type:           string
  chapter_id?:    number
  draft_id?:      number
  translation_id?: number
  stage?:         string
  page_index?:    number
  page_total?:    number
  error?:         string
}

/** Subscribe to live progress for one translation. Returns the latest
 *  snapshot; null until first event lands.
 *
 *  `enabled=false` short-circuits — useful while we wait for a spawn
 *  RPC to return the translation_id.
 */
export function useTranslateProgress(
  translationId: number | null | undefined,
  enabled = true,
): TranslateProgress | null {
  const [snap, setSnap] = useState<TranslateProgress | null>(null)

  useEffect(() => {
    if (!enabled) return
    const id = Number(translationId)
    if (!Number.isInteger(id) || id <= 0) return
    const token = getToken()
    if (!token) return

    const url = `${api.base}/api/translate/${id}/events`
                 + `?token=${encodeURIComponent(token)}`
    const es  = new EventSource(url)

    es.onmessage = (msg) => {
      let ev: RawEvent
      try { ev = JSON.parse(msg.data) } catch { return }
      setSnap((prev) => reduce(prev, ev))
    }

    es.onerror = () => {
      // EventSource auto-reconnects; we don't tear it down on error.
      // A terminal state (done/error) is communicated via a typed
      // StageDone/StageFailed event, not via the transport error.
    }

    return () => es.close()
  }, [translationId, enabled])

  return snap
}

function reduce(
  prev: TranslateProgress | null, ev: RawEvent,
): TranslateProgress | null {
  const base: TranslateProgress = prev ?? {
    state: 'pending', stage: '', index: 0, total: 0,
  }
  switch (ev.type) {
    case 'StageStarted':
      return { ...base, state: 'running', stage: ev.stage ?? base.stage }
    case 'PageDone':
      return {
        ...base,
        state: 'running',
        stage: ev.stage ?? base.stage,
        index: Number(ev.page_index ?? 0),
        total: Number(ev.page_total ?? 0),
      }
    case 'StageDone':
      // Render's StageDone is the terminal event — anything before
      // (scan, translate) just bumps the stage label and clears the
      // page counters in preparation for the next stage's PageDone.
      if (ev.stage === 'render') {
        return { ...base, state: 'done', stage: 'render' }
      }
      return {
        ...base, state: 'running', stage: ev.stage ?? base.stage,
        index: 0, total: 0,
      }
    case 'StageFailed':
      return {
        ...base, state: 'error', stage: ev.stage ?? base.stage,
        error: ev.error || 'Pipeline failed',
      }
    default:
      return base
  }
}
