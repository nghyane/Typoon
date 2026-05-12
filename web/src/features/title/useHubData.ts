// Hub data hook — merge DB chapters with manifest-live chapter list.
//
// A library entry's primary material may carry zero materialized
// chapter rows in the DB (chapters are lazy — only created when a
// user spawns a translation). The hub still needs to show the full
// chapter list from day one, so we fetch:
//
//   • /api/library/entry/{id}             — entry metadata
//   • /api/material/{primary_material_id} — DB chapters (with
//     translation overlay) for chapters that have been touched
//   • manifest.fetchMangaDetail            — live chapter list from
//     the source. Free egress via the DA proxy; no backend hop.
//
// Merge key: `upstream_url`. A manifest chapter with a matching DB
// chapter inherits the latter's translations + chapter id; the rest
// render as raw-only rows the user can `Dịch` to materialize.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api, type ApiChapter, type ApiChapterTranslation,
} from '@shared/api/api'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { useSources } from '@features/browse/sources'
import type {
  InstalledSource, MangaChapterRef, MangaDetail,
} from '@features/browse/manifest/types'
export interface HubChapterRow {
  /** Stable key for React lists. */
  key:           string
  /** DB chapter id when materialized; null when this row is manifest-only. */
  chapterId:     number | null
  number:        string
  label:         string | null
  upstreamUrl:   string | null
  position:      number
  pageCount:     number
  translations:  ApiChapterTranslation[]
  /** Whether the chapter exists in DB. Drives the 'Dịch' vs 'Đọc' affordance. */
  materialized:  boolean
}

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

  const installedSourcesMap = useSources((s) => s.sources)
  const sourceId = material.data?.material.source ?? null
  const installedSource: InstalledSource | null = useMemo(
    () => sourceId ? (installedSourcesMap[sourceId] ?? null) : null,
    [sourceId, installedSourcesMap],
  )

  const upstreamRef = material.data?.material.upstream_ref ?? null

  // Manifest-live chapter list. Only fires for source-backed materials
  // (origin='source' implies both source + upstream_ref are non-null).
  const manifest = useQuery({
    queryKey:  ['manifest', 'detail', installedSource?.manifest.id, upstreamRef],
    queryFn:   () => fetchMangaDetail(installedSource!.manifest, upstreamRef!),
    staleTime: 5 * 60_000,
    enabled:   installedSource !== null && upstreamRef !== null,
  })

  // Dev visibility — surface why the chapter list might be empty.
  if (import.meta.env.DEV && material.data && !manifest.isPending) {
    const m = material.data.material
    if (m.source && m.upstream_ref) {
      if (!installedSource) {
        // eslint-disable-next-line no-console
        console.warn(
          `[hub] no installed source matches material.source=${m.source}; `,
          `installed:`, Object.keys(installedSourcesMap),
        )
      } else if (manifest.error) {
        // eslint-disable-next-line no-console
        console.warn(
          `[hub] manifest fetch failed for ${m.source}:`,
          manifest.error,
        )
      } else if (manifest.data) {
        // eslint-disable-next-line no-console
        console.info(
          `[hub] manifest ${m.source} returned ${manifest.data.chapters.length} chapters`,
        )
      }
    }
  }

  const rows = useMemo(
    () => mergeChapters(material.data?.chapters ?? [], manifest.data ?? null, installedSource),
    [material.data?.chapters, manifest.data, installedSource],
  )

  return {
    entry:        entry.data ?? null,
    material:     material.data ?? null,
    rows,
    source:       installedSource,
    loading:      entry.isPending
                  || (primaryId !== null && material.isPending),
    chaptersLoading: installedSource !== null && upstreamRef !== null && manifest.isPending,
    error:        entry.error ?? material.error ?? manifest.error,
  }
}


/** Merge DB chapter rows with manifest-live ones.
 *
 *  • Manifest chapters drive the order (latest first by manifest).
 *  • A DB chapter joins by `upstream_url`; when found it contributes
 *    translations + chapter_id + page_count.
 *  • DB-only chapters (no manifest row — e.g. user-uploaded extras)
 *    append at the bottom in DB position order. */
function mergeChapters(
  dbChapters: ApiChapter[],
  detail:     MangaDetail | null,
  source:     InstalledSource | null,
): HubChapterRow[] {
  const byUrl = new Map<string, ApiChapter>()
  for (const c of dbChapters) {
    if (c.upstream_url) byUrl.set(c.upstream_url, c)
  }

  const rows: HubChapterRow[] = []
  const seenChapterIds = new Set<number>()

  if (detail) {
    detail.chapters.forEach((m: MangaChapterRef, i) => {
      const dbHit = byUrl.get(m.url)
      if (dbHit) seenChapterIds.add(dbHit.id)
      rows.push({
        key:           `m::${source?.manifest.id ?? '?'}::${m.id}`,
        chapterId:     dbHit?.id ?? null,
        number:        m.number,
        label:         m.label,
        upstreamUrl:   m.url,
        position:      -i,                // newest-first ordering hint
        pageCount:     dbHit?.page_count ?? 0,
        translations:  dbHit?.translations ?? [],
        materialized:  !!dbHit,
      })
    })
  }

  for (const c of dbChapters) {
    if (seenChapterIds.has(c.id)) continue
    rows.push({
      key:           `db::${c.id}`,
      chapterId:     c.id,
      number:        c.number,
      label:         c.label,
      upstreamUrl:   c.upstream_url,
      position:      c.position,
      pageCount:     c.page_count,
      translations:  c.translations,
      materialized:  true,
    })
  }

  return rows
}
