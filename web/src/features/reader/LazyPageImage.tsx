// LazyPageImage — resolves a per-page token lazily via React Query.
//
// When a raw source uses `resolvePageUrl` (e.g. E-Hentai showpage
// API), `ChapterPages.tokens` carries opaque tokens instead of real
// URLs. This component fires `usePageUrl` only when `inWindow` is
// true so the network call happens on viewport entry, not upfront.
//
// Cache: each token maps to its own React Query entry (staleTime:
// Infinity). Re-opening the same chapter or navigating back never
// re-fetches — the resolved URL is served from cache instantly.

import { usePageUrl } from './queries'
import { PageImage } from './PageImage'
import type { ReaderPage } from './types'
import type { InstalledSource } from '@features/browse/manifest/types'

interface Props {
  page:      ReaderPage
  source:    InstalledSource
  inWindow:  boolean
  className?: string
}

export function LazyPageImage({ page, source, inWindow, className }: Props) {
  const { data: resolvedUrl } = usePageUrl(source, page.token, inWindow)

  return (
    <PageImage
      page={page}
      src={resolvedUrl ?? null}
      inWindow={inWindow}
      className={className}
    />
  )
}
