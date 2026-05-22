// Single mutation surface for both "thêm vào thư viện" paths:
//   • importHit   — source-backed (useEnsureWorkFromSource + useAddToLibrary)
//   • importBlank — empty Work + library entry, no source seeded
//
// Both paths fold the snapshot into Dexie via the works/library hook
// stack, toast on success, and call the appropriate callback. Exposes
// plain callbacks (not raw useMutation) so consumers don't see React
// Query internals.

import { useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'

import {
  useEnsureWorkFromSource, useCreateBlankWork,
  type Work,
} from '@features/works/queries'
import { useAddToLibrary } from '@features/library/queries'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import type { MangaDetail } from '@features/browse/manifest/types'
import { toast } from '@shared/ui/Toaster'

import type { SearchHit } from './fanoutSearch'


const DEFAULT_TARGET_LANG  = 'vi'
const BLANK_TITLE_FALLBACK = 'Manga mới'


export interface ImportHitArgs {
  hit:     SearchHit
  /** Pre-fetched detail (URL-paste form has it in hand). When
   *  omitted the helper fetches it; on fetch failure it falls back
   *  to the hit snapshot rather than aborting. */
  detail?: MangaDetail | null
}


export interface ImportToLibrary {
  importHit:   (args: ImportHitArgs) => void
  importBlank: (title: string) => void
  isPending:   boolean
}


export function useImportToLibrary(
  opts: {
    onSuccess?:      (work: Work) => void
    /** Fires after a successful blank-create with the new Work so
     *  the caller can navigate the user to `/w/$id` and let them
     *  upload chapters right away. */
    onBlankCreated?: (work: Work) => void
  } = {},
): ImportToLibrary {
  const { onSuccess, onBlankCreated } = opts

  const ensure   = useEnsureWorkFromSource()
  const blank    = useCreateBlankWork()
  const addToLib = useAddToLibrary()

  const hitMut = useMutation({
    mutationFn: async ({ hit, detail }: ImportHitArgs): Promise<Work> => {
      // Resolve canonical detail (description, languages, …) so the
      // imported work carries richer metadata. Fall back to the hit
      // snapshot when the upstream call fails — a flaky source
      // should never block save.
      const resolved = detail
        ?? await fetchMangaDetail(hit.source.manifest, hit.manga.url)
             .catch(() => null)

      const manifest = hit.source.manifest
      const work = await ensure.mutateAsync({
        source:       manifest.id,
        upstream_ref: hit.manga.url,
        snapshot: {
          title:       resolved?.title    ?? hit.manga.title,
          cover_url:   resolved?.cover    ?? hit.manga.cover ?? null,
          source_lang: manifest.languages?.[0] ?? 'ja',
          target_lang: DEFAULT_TARGET_LANG,
          nsfw:        !!manifest.nsfw,
          languages:   resolved?.availableLanguages
                    ?? manifest.languages
                    ?? [],
        },
      })

      if (!work.in_library) {
        await addToLib.mutateAsync({ work_id: work.id })
      }
      return work
    },
    onSuccess: (work) => {
      toast.success(`Đã thêm "${work.title}" vào thư viện`)
      onSuccess?.(work)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const blankMut = useMutation({
    mutationFn: async (title: string): Promise<Work> => {
      const trimmed = title.trim() || BLANK_TITLE_FALLBACK
      const work = await blank.mutateAsync({
        title:       trimmed,
        target_lang: DEFAULT_TARGET_LANG,
      })
      await addToLib.mutateAsync({ work_id: work.id })
      return work
    },
    onSuccess: (work) => {
      toast.success('Đã tạo truyện mới. Tải chương đầu tiên ngay.')
      onBlankCreated?.(work)
      onSuccess?.(work)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return {
    importHit:   useCallback((a: ImportHitArgs) => hitMut.mutate(a), [hitMut]),
    importBlank: useCallback((t: string)        => blankMut.mutate(t), [blankMut]),
    isPending:   hitMut.isPending || blankMut.isPending,
  }
}
