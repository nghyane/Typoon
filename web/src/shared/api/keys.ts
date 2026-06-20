// React Query key factory — v3.5 surface.
//
// Server-backed:
//   session, context
// Client-backed (Dexie, queried via TanStack Query for reactivity):
//   works (resolution + metadata)
//   library (pin/promotion layer over works)
//   history, localSettings
// Source-adapter (live HTTP via CORS proxy):
//   manifest (search, detail, chapterPages, pageUrl)

import type { LibraryStatus } from '@shared/db'

export const qk = {
  session: {
    self:    ()                   => ['session'] as const,
    config:  ()                   => ['session', 'config'] as const,
  },

  works: {
    byId:        (workId: string) => ['works', workId] as const,
    bySourceRef: (
      source:       string | null | undefined,
      upstream_ref: string | null | undefined,
    ) => ['works', 'by-source-ref', source ?? null, upstream_ref ?? null] as const,
    recent:      ()               => ['works', 'recent'] as const,
    invalid:     ()               => ['works', 'invalid'] as const,
  },

  library: {
    all:       ()                       => ['library'] as const,
    byStatus:  (status: LibraryStatus)  => ['library', 'by-status', status] as const,
    history:   (workId: string)         => ['library', workId, 'history'] as const,
  },

  history: {
    all:     ()                   => ['history'] as const,
    forWork: (workId: string)     => ['history', workId] as const,
  },

  localSettings:  ()              => ['settings', 'local'] as const,
  publicSettings: ()              => ['settings', 'public'] as const,

  // Source adapters — same convention as before so existing browse-mode
  // code keeps working.
  manifest: {
    detail:        (sourceId: string | null | undefined,
                    upstreamRef: string | null | undefined) =>
      ['manifest', 'detail', sourceId ?? null, upstreamRef ?? null] as const,
    chapterPages:  (sourceId: string, chapterUrl: string) =>
      ['manifest', 'chapter-pages', sourceId, chapterUrl] as const,
    pageUrl:       (sourceId: string, token: string) =>
      ['manifest', 'page-url', sourceId, token] as const,
    search:        (sourceId: string, q: string) =>
      ['manifest', 'search', sourceId, q] as const,
  },
} as const
