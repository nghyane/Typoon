// Hub data hook — fetch all linked materials + their manifest chapter
// lists in parallel, then fold into a cross-material chapter view.
//
// One library entry can link multiple materials (each its own source,
// possibly different raw languages). We hit:
//
//   • /api/library/entry/{id}                       — entry shape
//   • /api/material/{id}      for every link         — DB chapters
//                                                       + translation overlay
//   • manifest.fetchMangaDetail for source-backed   — live raw chapter list
//
// Each level uses keepPreviousData so a switch between entries or a
// translation overlay refresh doesn't flicker the chapter list.

import { useMemo } from 'react'
import { useQuery, useQueries, keepPreviousData } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { useSources } from '@features/browse/sources'
import type { InstalledSource } from '@features/browse/manifest/types'
import {
  mergeChapters, type MaterialBundle, type HubChapter,
} from './mergeChapters'


export function useHubData(entryId: number) {
  const entry = useQuery({
    queryKey:         ['library', 'entry', entryId],
    queryFn:          () => api.getLibraryEntry(entryId),
    staleTime:        30_000,
    enabled:          Number.isFinite(entryId) && entryId > 0,
    placeholderData:  keepPreviousData,
  })

  // Linked material ids in entry-stable order (primary first when
  // present, then others by linked_at). We rely on the API to return
  // materials in this order; if it doesn't, the merge result still
  // works — only the version ordering inside a chapter changes.
  const materialIds = useMemo(() => {
    const ids = (entry.data?.materials ?? []).map((m) => m.material_id)
    const primary = entry.data?.primary_material_id
    if (primary != null) {
      // Hoist primary to the front so its versions surface first
      // inside each chapter row.
      const rest = ids.filter((id) => id !== primary)
      return [primary, ...rest]
    }
    return ids
  }, [entry.data])

  // /api/material/{id} for every linked material — parallel.
  const materialQueries = useQueries({
    queries: materialIds.map((id) => ({
      queryKey:         ['material', 'detail', id],
      queryFn:          () => api.getMaterial(id),
      staleTime:        30_000,
      placeholderData:  keepPreviousData,
    })),
  })

  // Installed source registry — feeds the manifest fetches below.
  const installed = useSources((s) => s.sources)

  // For each loaded material, fire manifest.fetchMangaDetail when
  // it's source-backed and the source is installed.
  const manifestQueries = useQueries({
    queries: materialQueries.map((q) => {
      const m   = q.data?.material
      const src = m?.source ? (installed[m.source] ?? null) : null
      return {
        queryKey: ['manifest', 'detail', src?.manifest.id, m?.upstream_ref],
        queryFn:  () => fetchMangaDetail(src!.manifest, m!.upstream_ref!),
        staleTime:        5 * 60_000,
        enabled:          src !== null && !!m?.upstream_ref,
        retry:            false,
        placeholderData:  keepPreviousData,
      }
    }),
  })

  // Build bundles in materialIds order — keep manifests aligned with
  // their material by index.
  const bundles: MaterialBundle[] = useMemo(() => {
    const out: MaterialBundle[] = []
    for (let i = 0; i < materialQueries.length; i++) {
      const detail = materialQueries[i]!.data
      if (!detail) continue
      const sourceId = detail.material.source
      const source: InstalledSource | null = sourceId
        ? (installed[sourceId] ?? null)
        : null
      out.push({
        detail,
        source,
        manifest: manifestQueries[i]?.data ?? null,
      })
    }
    return out
  }, [materialQueries, manifestQueries, installed])

  const chapters: HubChapter[] = useMemo(
    () => mergeChapters(bundles),
    [bundles],
  )

  const primaryMaterial = materialQueries.find(
    (q) => q.data?.material.id === entry.data?.primary_material_id,
  )?.data ?? null

  const anyMaterialPending = materialQueries.some((q) => q.isPending)
  const anyManifestPending = manifestQueries.some((q) => q.isPending && q.fetchStatus !== 'idle')
  const anyManifestError   = manifestQueries.some((q) => q.error)

  return {
    entry:           entry.data ?? null,
    /** Convenience: primary material for the hero render. */
    primaryMaterial,
    /** Every loaded material — used by power features (Sources tab,
     *  material picker, link / unlink). */
    materials:       materialQueries.map((q) => q.data).filter((m): m is NonNullable<typeof m> => !!m),
    chapters,

    loading:         entry.isPending || (materialIds.length > 0 && anyMaterialPending),
    chaptersLoading: anyManifestPending,
    chaptersError:   anyManifestError,
    error:           entry.error,
  }
}

// Re-export the row shape so callers can type their renderers without
// digging into the merge module.
export type { HubChapter, HubVersion } from './mergeChapters'
