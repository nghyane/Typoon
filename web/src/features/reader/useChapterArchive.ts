// Stream a chapter archive (BNL) — one HTTP request, RAF-batched updates.
//
// Cancellation flows through a single AbortController; aborting on
// cleanup propagates to:
//   • Bunle.open()'s Range request (index fetch)
//   • Bunle.stream()'s full-body fetch + reader.cancel()
// so unmounting mid-load stops bandwidth within a single packet.
//
// Why ref-based urls: stream() yields N blobs (~1/s each). setState
// per blob causes O(N) reconciles; with 60 slots that's O(N²) diff
// work. We mutate a Map in a ref and bump a single tick state once
// per animation frame so React reconciles at most once per ~16ms.

import { useEffect, useRef, useState } from 'react'
import { Bunle } from '@nghyane/bunle'

export interface ChapterArchive {
  bunle:    Bunle | null
  /** index → blob: URL. Mutates in place; read after each tick bump. */
  urls:     ReadonlyMap<number, string>
  progress: { received: number; total: number }
  loading:  boolean
  error:    string | null
}

export function useChapterArchive(url: string | null | undefined): ChapterArchive {
  const [bunle, setBunle]       = useState<Bunle | null>(null)
  const [progress, setProgress] = useState({ received: 0, total: 0 })
  const [error, setError]       = useState<string | null>(null)
  // Tick exists only to trigger reconcile when urls Map mutates.
  // Components read urls via the returned reference — the Map itself
  // is stable; we bump the tick to signal "look again".
  const [, setTick] = useState(0)

  const urlsRef = useRef<Map<number, string>>(new Map())

  useEffect(() => {
    if (!url) {
      setBunle(null)
      setProgress({ received: 0, total: 0 })
      setError(null)
      urlsRef.current = new Map()
      return
    }

    const ctrl = new AbortController()
    let opened: Bunle | null = null
    const localUrls: string[] = []

    // RAF-batched tick — many blob arrivals → one reconcile per frame.
    let rafScheduled = false
    const scheduleTick = () => {
      if (rafScheduled || ctrl.signal.aborted) return
      rafScheduled = true
      requestAnimationFrame(() => {
        rafScheduled = false
        if (!ctrl.signal.aborted) setTick((t) => t + 1)
      })
    }

    setError(null)
    setBunle(null)
    setProgress({ received: 0, total: 0 })
    urlsRef.current = new Map()

    ;(async () => {
      try {
        const b = await Bunle.open(url, { signal: ctrl.signal })
        if (ctrl.signal.aborted) { b.close(); return }
        opened = b
        setBunle(b)

        for await (const { index, blob } of b.stream({
          signal: ctrl.signal,
          onProgress: (received, total) => {
            if (ctrl.signal.aborted) return
            setProgress({ received, total })
          },
        })) {
          const u = URL.createObjectURL(blob)
          localUrls.push(u)
          urlsRef.current.set(index, u)
          scheduleTick()
        }
      } catch (e) {
        // AbortError on cleanup is expected — not a real failure.
        if (ctrl.signal.aborted) return
        setError(e instanceof Error ? e.message : String(e))
      }
    })()

    return () => {
      ctrl.abort()
      // Revoke every URL we created in this run. `bunle.close()` only
      // knows about URLs allocated via `bunle.url(i)`; stream blobs are
      // ours to manage.
      for (const u of localUrls) URL.revokeObjectURL(u)
      if (opened) opened.close()
    }
  }, [url])

  return {
    bunle,
    urls: urlsRef.current,
    progress,
    loading: !!url && bunle === null && error === null,
    error,
  }
}
