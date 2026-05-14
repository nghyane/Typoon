// Single mutation surface for both "thêm vào thư viện" paths:
//   • importHit   — source-backed (importMaterialFromHit + library entry)
//   • importBlank — empty Work + library entry, no material seeded
//
// Defaults (vi, reading) are overwritable on the Work hub. Both paths
// invalidate the library cache, toast, and call the appropriate
// success callback. Exposes plain callbacks (not raw useMutation) so
// consumers don't see React Query internals.
//
// Material-creation logic for the source-backed path is delegated to
// `@features/material/import` — that module is the ONLY place that
// builds the wire payload for a source-backed material. Adding a
// field to the import schema is a single-file edit there.
//
// The blank path goes through `POST /api/work` which creates the
// Work + library entry server-side without seeding a placeholder
// "upload" material. The viewer's upload material is lazy-created on
// the first chapter upload (`POST /api/work/{id}/upload-init`), so
// users who only ever follow source-backed manga don't accumulate
// empty material rows.

import { useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api, type ApiWorkDetail } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { toast } from '@shared/ui/Toaster'
import { importMaterialFromHit } from '@features/material/import'
import type { MangaDetail } from '@features/browse/manifest/types'
import type { SearchHit } from './fanoutSearch'


const DEFAULT_TARGET_LANG  = 'vi'
const DEFAULT_STATUS       = 'reading' as const
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
    onSuccess?:      () => void
    /** Fires after a successful blank-create with the new Work id
     *  so the caller can navigate the user to `/w/$id` and let them
     *  upload chapters right away. */
    onBlankCreated?: (work: ApiWorkDetail) => void
  } = {},
): ImportToLibrary {
  const qc = useQueryClient()
  const { onSuccess, onBlankCreated } = opts

  const hitMut = useMutation({
    mutationFn: async ({ hit, detail }: ImportHitArgs) => {
      const material = await importMaterialFromHit(hit, detail)
      // Title + cover are NOT sent — they live on the material we
      // just imported and the server resolves the entry's display
      // fields from there at read time. The toast/onSuccess
      // callback uses the material's title locally for feedback.
      await api.createLibraryEntry({
        material_id: material.id,
        target_lang: DEFAULT_TARGET_LANG,
        status:      DEFAULT_STATUS,
      })
      return material.title
    },
    onSuccess: (title) => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      toast.success(`Đã thêm "${title}" vào thư viện`)
      onSuccess?.()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const blankMut = useMutation({
    mutationFn: async (title: string) => {
      const trimmed = title.trim() || BLANK_TITLE_FALLBACK
      return api.createBlankWork({
        title:       trimmed,
        target_lang: DEFAULT_TARGET_LANG,
      })
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      toast.success('Đã tạo truyện mới. Tải chương đầu tiên ngay.')
      onBlankCreated?.(work)
      onSuccess?.()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return {
    importHit:   useCallback((a: ImportHitArgs) => hitMut.mutate(a), [hitMut]),
    importBlank: useCallback((t: string)        => blankMut.mutate(t), [blankMut]),
    isPending:   hitMut.isPending || blankMut.isPending,
  }
}
