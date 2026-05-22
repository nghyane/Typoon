// URL → source detection.
//
// User pastes a manga URL in the AddMangaModal; we figure out which
// installed source owns the hostname and turn the URL into the
// `(source, upstream_ref)` pair the work resolver expects.
//
// When no installed source matches the hostname, the modal falls
// back to "tạo trống". The `upstream_ref` is the URL itself — the
// manifest runtime takes that as the input to `fetchMangaDetail`,
// so anything else would just add a re-derivation step.

import type { InstalledSource } from '@features/browse/manifest/types'

export interface PasteMatch {
  source:      InstalledSource
  /** What we feed `fetchMangaDetail` + `useEnsureWorkFromSource` as upstream_ref. */
  upstreamRef: string
}

const URL_RE = /^https?:\/\//i

export function isUrlLike(input: string): boolean {
  return URL_RE.test(input.trim())
}

/** Resolve a pasted URL to an installed source by hostname match.
 *  Returns null when none of the installed sources own this host. */
export function matchSource(
  raw:     string,
  sources: InstalledSource[],
): PasteMatch | null {
  const trimmed = raw.trim()
  if (!isUrlLike(trimmed)) return null

  let parsed: URL
  try { parsed = new URL(trimmed) }
  catch { return null }

  // Match by exact host first, then by suffix (so `www.x.com` resolves
  // to a source declaring `x.com`). The proxy allow-list keys on
  // `host` so the registry guarantees one is present per source.
  const wanted = parsed.host.toLowerCase()
  const direct = sources.find(
    s => s.enabled && s.manifest.host.toLowerCase() === wanted,
  )
  const suffix = direct ?? sources.find(
    s => s.enabled
      && wanted.endsWith('.' + s.manifest.host.toLowerCase()),
  )
  if (!suffix) return null

  return { source: suffix, upstreamRef: trimmed }
}
