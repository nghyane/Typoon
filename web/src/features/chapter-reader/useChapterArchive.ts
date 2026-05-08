import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Bunle } from '@nghyane/bunle'
import { api } from '@shared/api/api'

interface Result {
  bunle:   Bunle | null
  loading: boolean
  error:   string | null
}

/**
 * Two-step archive opening:
 *   1. fetch a short-lived signed URL (auth-required, cached by RQ)
 *   2. open the Bunle archive at that URL (public Range requests)
 *
 * The signed URL is the cache key for browsers, CDN edges, and R2 — no
 * Authorization header on the heavy bytes. Rotates every 15 min server-side.
 *
 * The Bunle instance owns object URL caches; we close it on unmount or
 * before swapping to a new chapter.
 */
export function useChapterArchive(projectId: number, chapterId: number): Result {
  const [bunle, setBunle] = useState<Bunle | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { data: signed, isPending: signedPending, error: signedError } = useQuery({
    queryKey: ['render-url', projectId, chapterId],
    queryFn:  () => api.getRenderArchive(projectId, chapterId),
    enabled:  !isNaN(projectId) && !isNaN(chapterId),
    // The URL is valid for 15 min server-side. Refresh slightly before
    // that to avoid expiring mid-read.
    staleTime: 12 * 60_000,
  })

  useEffect(() => {
    if (!signed) return
    let cancelled = false
    let opened: Bunle | null = null

    setError(null)
    setBunle(null)

    Bunle.open(signed.url)
      .then((b) => {
        if (cancelled) { b.close(); return }
        opened = b
        setBunle(b)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
      })

    return () => {
      cancelled = true
      if (opened) opened.close()
    }
  }, [signed])

  return {
    bunle,
    loading: signedPending || (!!signed && bunle === null && error === null),
    error:   error ?? (signedError as Error | null)?.message ?? null,
  }
}
