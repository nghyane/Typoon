// React Query key factory — single source of truth for every cache
// entry in the SPA.
//
// Why a factory:
//   1. Grep-friendly. Every cache lookup goes through one symbol;
//      finding "who reads / who invalidates the work cache" is one
//      search instead of grepping for `['work',`.
//   2. Type safety. The `as const` tuples preserve literal types so
//      TanStack Query's TS inference picks them up without manual
//      generics.
//   3. Refactor safety. Changing a key (e.g. adding a dimension to
//      the manifest cache) is one edit here, not a sprawl across the
//      codebase.
//
// Conventions:
//   • Top-level segment groups by domain (work, library, manifest…).
//   • Add narrowing segments left-to-right (broader → narrower) so
//     `invalidateQueries(qk.work.all)` matches every work entry,
//     `invalidateQueries(qk.work.byId(id))` only one.

export const qk = {
  // ── Server domain payloads ──────────────────────────────────
  work: {
    all:           ()                  => ['work'] as const,
    byId:          (workId: number)    => ['work', workId] as const,
    linkSuggest:   (workId: number)    => ['work', workId, 'link-suggestions'] as const,
  },

  translation: {
    byId:          (id: number)        => ['translation', id] as const,
  },

  library: {
    all:           ()                  => ['library'] as const,
  },

  workers:         ()                  => ['workers'] as const,
  quota:           ()                  => ['quota'] as const,
  tokens:          ()                  => ['tokens'] as const,

  me: {
    self:          ()                  => ['me'] as const,
    recentReads:   ()                  => ['me', 'recent-reads'] as const,
  },

  community: {
    recent:        ()                  => ['community', 'recent'] as const,
  },

  // ── Source manifest (client-side fetched, treated like server) ─
  manifest: {
    detail:        (sourceId: string | null | undefined,
                    upstreamRef: string | null | undefined) =>
      ['manifest', 'detail', sourceId ?? null, upstreamRef ?? null] as const,
    chapterPages:  (sourceId: string, chapterUrl: string) =>
      ['manifest', 'chapter-pages', sourceId, chapterUrl] as const,
    search:        (sourceId: string, q: string) =>
      ['manifest', 'search', sourceId, q] as const,
  },
} as const
