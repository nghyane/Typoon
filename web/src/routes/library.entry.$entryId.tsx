import { createFileRoute, redirect } from '@tanstack/react-router'
import { api } from '@shared/api/api'

// =============================================================================
// /library/entry/$entryId — resolver that fans out to the right route.
//
// The library card link is generic: clicking just picks "the manga
// I'm following". This loader resolves the entry's primary material,
// looks up its source + upstream_ref, and redirects to the correct
// /browse/$source/manga/$mangaId path (which has the rich detail UI).
//
// We chose route-level resolution over Link-time URL synthesis so the
// library card stays card-shaped and doesn't need a per-entry fetch.
// Loaders run once per click; React Query caches the entry list so
// no extra round-trip on most navigations.
// =============================================================================

export const Route = createFileRoute('/library/entry/$entryId')({
  loader: async ({ params }) => {
    const entryId = Number(params.entryId)
    if (!Number.isInteger(entryId) || entryId <= 0) {
      throw redirect({ to: '/library' })
    }
    const entry = await api.getLibraryEntry(entryId).catch(() => null)
    if (entry == null || entry.primary_material_id == null) {
      throw redirect({ to: '/library' })
    }
    const detail = await api.getMaterial(entry.primary_material_id)
      .catch(() => null)
    if (detail == null || detail.material.source == null
        || detail.material.upstream_ref == null) {
      // Ext / upload materials don't have a manifest-source detail
      // page yet — back to library.
      throw redirect({ to: '/library' })
    }
    throw redirect({
      to: '/browse/$source/manga/$mangaId',
      params: {
        source:  detail.material.source,
        mangaId: encodeURIComponent(detail.material.upstream_ref),
      },
    })
  },
  // Empty component — the loader always redirects.
  component: () => null,
})
