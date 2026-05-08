import { useEffect, useState } from 'react'
import { Bunle } from '@nghyane/bunle'

interface Result {
  bunle:   Bunle | null
  loading: boolean
  error:   string | null
}

/**
 * Open a chapter's rendered archive from its public CDN URL.
 *
 * The URL is supplied by the API in `chapter.archive_url`; it points at
 * a public CDN object whose path is an HMAC token of (project, chapter).
 * No auth header is needed — the path itself is the capability. The
 * version query string (`?v=updated_at`) busts the CDN cache when a
 * chapter re-renders; the path stays stable for cache key purposes.
 *
 * The Bunle instance owns object-URL caches; we close it on unmount
 * or before swapping to a new chapter.
 */
export function useChapterArchive(url: string | null | undefined): Result {
  const [bunle, setBunle] = useState<Bunle | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!url) {
      setBunle(null)
      setError(null)
      return
    }
    let cancelled = false
    let opened: Bunle | null = null

    setError(null)
    setBunle(null)

    Bunle.open(url)
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
  }, [url])

  return {
    bunle,
    loading: !!url && bunle === null && error === null,
    error,
  }
}
