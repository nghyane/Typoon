// Hub data hook — assemble the title detail page for a library entry.
//
// Phase 1 strategy:
//   • Read /api/library/entry/{id} for the entry shape + linked
//     materials list.
//   • Read /api/material/{primary_material_id} for chapters +
//     per-chapter translation overlay (the backend already embeds
//     these in the detail response).
//
// Cross-material chapter merge lands in a follow-up slice — for
// now the hub shows the primary material's chapter list directly.
// The entry-level metadata (status, target_lang, translation
// summary) wraps it.

import { useQuery } from '@tanstack/react-query'
import { api } from '@shared/api/api'

export function useHubData(entryId: number) {
  const entry = useQuery({
    queryKey:  ['library', 'entry', entryId],
    queryFn:   () => api.getLibraryEntry(entryId),
    staleTime: 30_000,
    enabled:   Number.isFinite(entryId) && entryId > 0,
  })

  const primaryId = entry.data?.primary_material_id ?? null

  const material = useQuery({
    queryKey:  ['material', 'detail', primaryId],
    queryFn:   () => api.getMaterial(primaryId!),
    staleTime: 30_000,
    enabled:   primaryId !== null,
  })

  return {
    entry:        entry.data ?? null,
    material:     material.data ?? null,
    loading:      entry.isPending || (primaryId !== null && material.isPending),
    error:        entry.error ?? material.error,
  }
}
