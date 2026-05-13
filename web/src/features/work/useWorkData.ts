// useWorkData — composed view of the Work page.
//
// Built on top of domain queries (`useWork`, `useMangaDetail`) so
// the cache layer is shared with every other surface that needs the
// same data (the reader, the in-progress section). This hook adds
// the UI-shaped derivations on top: active material resolution,
// merged HubChapter list, source plugin look-ups.

import { useMemo } from 'react'

import type { ApiMaterial, ApiWorkDetail } from '@shared/api/api'
import { useSources } from '@features/browse/sources'
import {
  mergeChapters, type HubChapter,
} from '@features/title/mergeChapters'

import { useWork, useMangaDetail } from './queries'


export interface WorkData {
  work:             ApiWorkDetail | null
  materials:        ApiMaterial[]
  activeMaterial:   ApiMaterial | null
  targetLang:       string | null
  chapters:         HubChapter[]
  workLoading:      boolean
  manifestLoading:  boolean
  workError:        Error | null
  manifestError:    Error | null
}


export function useWorkData(
  workId:        number,
  preferredSrc?: number | null,
): WorkData {
  const installed = useSources((s) => s.sources)

  const workQ = useWork(workId)
  const work = workQ.data ?? null
  const materials = work?.materials ?? []

  const activeMaterial = useMemo<ApiMaterial | null>(() => {
    if (materials.length === 0) return null
    if (preferredSrc != null) {
      const hit = materials.find((m) => m.id === preferredSrc)
      if (hit) return hit
    }
    // Default: oldest material that we have an installed source for
    // (prevents picking a sibling whose plugin isn't installed → empty
    // chapter spine). Fallback to materials[0] when none is installed.
    const withSource = materials.find(
      (m) => m.source != null && installed[m.source] != null,
    )
    return withSource ?? materials[0]!
  }, [materials, preferredSrc, installed])

  const activeSource = activeMaterial?.source
    ? (installed[activeMaterial.source] ?? null)
    : null

  // Reading-lang preference lives on `viewer_entry.target_lang` once
  // the user has bookmarked the Work. Drives chapter list overlay
  // badges + the spawn-fallback target.
  const targetLang = work?.viewer_entry?.target_lang ?? null

  const manifestQ = useMangaDetail(
    activeSource,
    activeMaterial?.upstream_ref ?? null,
  )

  const chapters = useMemo<HubChapter[]>(() => {
    if (!work) return []
    return mergeChapters({
      work,
      activeMaterialId: activeMaterial?.id ?? null,
      manifestChapters: manifestQ.data?.chapters ?? [],
      activeSource,
      installedSources: installed,
    })
  }, [work, activeMaterial, manifestQ.data, activeSource, installed])

  return {
    work,
    materials,
    activeMaterial,
    targetLang,
    chapters,
    workLoading:     workQ.isPending,
    manifestLoading: manifestQ.isPending && manifestQ.fetchStatus !== 'idle',
    workError:       (workQ.error as Error | null) ?? null,
    manifestError:   (manifestQ.error as Error | null) ?? null,
  }
}
