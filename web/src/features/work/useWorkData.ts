// useWorkData — composed view of the Work page.
//
// Built on top of domain queries (`useWork`, `useMangaDetail`) so the
// cache layer is shared with every other surface that needs the same
// data (the reader, the in-progress section).
//
// After community-vote Work merges, one Work may carry N sibling
// materials backed by different sources. This hook fetches every
// installed-source manifest IN PARALLEL via `useQueries` and unions
// the resulting chapter spines into a single `HubChapter[]`. There
// is no "active source" concept any more: the chapter list shows
// every readable version (raw + translation) across every source.
//
// Failure mode is silent skip — a manifest fetch that fails (plugin
// uninstalled, upstream down) contributes no raws but doesn't break
// the page; other sources keep rendering.
//
// Cache keys mirror `useMangaDetail` so a `useQuery(qk.manifest.detail
// (sourceId, ref))` somewhere else (e.g. the reader page-loader)
// shares the same entry. The persistence layer
// (`shared/api/persistence.ts`) already opts `manifest` keys into
// IndexedDB rehydration; nothing extra to wire up.

import { useMemo } from 'react'
import { keepPreviousData, useQueries } from '@tanstack/react-query'

import type { ApiMaterial, ApiWorkDetail } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { useSources } from '@features/browse/sources'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { useSession } from '@features/auth/session'
import { resolveReadingLang } from '@features/auth/readingLang'
import {
  mergeChapters, type HubChapter, type ManifestSource,
} from '@features/title/mergeChapters'

import { useWork } from './queries'


export interface WorkData {
  work:             ApiWorkDetail | null
  materials:        ApiMaterial[]
  /** Resolved reading-lang: per-Work override → user default →
   *  FALLBACK_LANG. Always a string. */
  targetLang:       string
  chapters:         HubChapter[]
  workLoading:      boolean
  /** True while ANY source-material manifest is in-flight. The
   *  chapter list keeps re-rendering as each manifest lands; callers
   *  use this for a list-level "loading…" hint, not to gate the
   *  whole page. */
  manifestsLoading: boolean
  workError:        Error | null
}


export function useWorkData(workId: number): WorkData {
  const installed = useSources((s) => s.sources)

  const workQ = useWork(workId)
  const work = workQ.data ?? null
  // Stable ref: `work?.materials ?? []` produces a new empty array
  // every render when `work` is null, which would invalidate every
  // downstream useMemo that depends on it. Memoise on `work` so the
  // empty case shares one frozen reference.
  const materials = useMemo<ApiMaterial[]>(
    () => work?.materials ?? [],
    [work],
  )

  // Reading-lang: per-Work `library_entries.target_lang` overrides
  // the viewer-wide `users.preferred_target_lang`; both fall back to
  // FALLBACK_LANG when missing. Same chain the reader uses, so the
  // hub badge and the reader's chosen version stay in sync.
  const { user } = useSession()
  const targetLang = resolveReadingLang(
    work?.viewer_entry?.target_lang,
    user?.preferred_target_lang,
  )

  // Every material whose source plugin is installed AND that has an
  // upstream_ref to feed the manifest fetch. Ext / upload materials
  // (source === null) and source-materials with no upstream_ref are
  // skipped — they contribute no manifest chapters by definition.
  const sourceMaterials = useMemo(
    () => materials.filter(
      (m) => m.source != null
          && installed[m.source] != null
          && !!m.upstream_ref,
    ),
    [materials, installed],
  )

  // Parallel manifest fetches. Each sub-query is keyed on
  // (source, upstream_ref) so it shares cache with `useMangaDetail`
  // anywhere else in the app. Failures are isolated: one source going
  // dark doesn't block the rest of the chapter list from rendering.
  const manifestQs = useQueries({
    queries: sourceMaterials.map((m) => ({
      queryKey: qk.manifest.detail(m.source, m.upstream_ref),
      queryFn:  () => fetchMangaDetail(installed[m.source!]!.manifest, m.upstream_ref!),
      staleTime: 5 * 60_000,
      gcTime:    24 * 60 * 60_000,
      retry:     2,
      retryDelay: (attempt: number) => Math.min(1000 * 2 ** attempt, 8000),
      placeholderData: keepPreviousData,
    })),
  })

  // Fold ready manifests into the merge input. RQ keeps each
  // sub-query's `data` referentially stable while the cache entry is
  // unchanged, so this folds to the same array across re-renders that
  // didn't actually land new data \u2014 cheap enough to do every render
  // without memo gymnastics.
  const manifestSources: ManifestSource[] = []
  for (let i = 0; i < sourceMaterials.length; i++) {
    const data = manifestQs[i]?.data
    if (!data) continue                         // silent skip
    const m = sourceMaterials[i]!
    manifestSources.push({
      material: m,
      source:   installed[m.source!]!,
      chapters: data.chapters,
    })
  }

  const chapters = useMemo<HubChapter[]>(() => {
    if (!work) return []
    return mergeChapters({
      work,
      manifestSources,
      installedSources: installed,
    })
    // `manifestSources` is rebuilt every render; depend on the data
    // payloads it folds (RQ keeps each entry referentially stable).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [work, installed, ...manifestQs.map((q) => q.data)])

  const manifestsLoading = manifestQs.some(
    (q) => q.isPending && q.fetchStatus !== 'idle',
  )

  return {
    work,
    materials,
    targetLang,
    chapters,
    workLoading:      workQ.isPending,
    manifestsLoading,
    workError:        (workQ.error as Error | null) ?? null,
  }
}
