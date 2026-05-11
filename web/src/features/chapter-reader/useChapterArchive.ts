import { useEffect, useState } from 'react'
import { Bunle } from '@nghyane/bunle'

interface Result {
  bunle:    Bunle | null
  /** index → object URL. Filled progressively as bytes arrive. */
  urls:     Record<number, string>
  /** bytes received / total content-length (0 if unknown). */
  progress: { received: number; total: number }
  loading:  boolean
  error:    string | null
}

/**
 * Open a chapter's rendered archive and stream every page in a single
 * HTTP request.
 *
 * Bunle exposes two access modes:
 *   - `bunle.url(i)` / `bunle.blob(i)` — one Range request per page.
 *   - `bunle.stream()`                 — one full GET, yields each page
 *                                        blob the moment its last byte
 *                                        arrives.
 *
 * For a manga reader the user reads sequentially, so per-page Range
 * requests are pure overhead (HTTP headers × N, TLS contention, no
 * progressive rendering). The Bunle docs are explicit: galleries with
 * 5+ sequential images should use `stream()` — one connection, zero
 * overhead, images appear progressively.
 *
 * Network shape per chapter:
 *   1. `Bunle.open(url)`  — Range bytes 0..index_size, gives every
 *                           page's width/height for instant layout.
 *   2. `bunle.stream()`   — single GET for the rest of the file.
 *
 * The Bunle instance owns the object-URL cache; we revoke everything on
 * unmount or before swapping to a new chapter so a `blob:` URL can never
 * outlive its archive.
 */
export function useChapterArchive(url: string | null | undefined): Result {
  const [bunle, setBunle]       = useState<Bunle | null>(null)
  const [urls, setUrls]         = useState<Record<number, string>>({})
  const [progress, setProgress] = useState({ received: 0, total: 0 })
  const [error, setError]       = useState<string | null>(null)

  useEffect(() => {
    if (!url) {
      setBunle(null)
      setUrls({})
      setProgress({ received: 0, total: 0 })
      setError(null)
      return
    }

    let cancelled = false
    let opened: Bunle | null = null
    const localUrls: string[] = []

    setError(null)
    setBunle(null)
    setUrls({})
    setProgress({ received: 0, total: 0 })

    ;(async () => {
      try {
        const b = await Bunle.open(url)
        if (cancelled) { b.close(); return }
        opened = b
        setBunle(b)

        for await (const { index, blob } of b.stream({
          onProgress: (received, total) => {
            if (cancelled) return
            setProgress({ received, total })
          },
        })) {
          if (cancelled) return
          const u = URL.createObjectURL(blob)
          localUrls.push(u)
          setUrls((prev) => ({ ...prev, [index]: u }))
        }
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      }
    })()

    return () => {
      cancelled = true
      // Revoke URLs we created in this effect run. `bunle.close()` would
      // only revoke URLs allocated through `bunle.url(i)`; ours came from
      // `stream()` blobs and are owned here.
      for (const u of localUrls) URL.revokeObjectURL(u)
      if (opened) opened.close()
    }
  }, [url])

  return {
    bunle,
    urls,
    progress,
    loading: !!url && bunle === null && error === null,
    error,
  }
}
