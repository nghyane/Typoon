// WorkActionsContext — stable mutation handlers bound to the current
// work id. Hoisted out of identity/chapter contexts so a mutation
// doesn't invalidate display contexts.
//
// Pattern: each handler is wrapped in `useCallback` keyed only on
// `workId` + the (stable) `mutate` fn from TanStack Query. The
// context value memo therefore only re-creates when one of those
// changes — effectively never within a single work-page mount.

import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react'

import {
  useUpdateWork, useAttachSource, useDetachSource,
  type WorkSource,
} from '@features/works/queries'
import {
  useAddToLibrary, useRemoveFromLibrary, useUpdateLibraryStatus,
} from '@features/library/queries'
import type { LibraryStatus } from '@shared/db'
import { Bunle, type PackInput } from '@nghyane/bunle'
import { useMutation } from '@tanstack/react-query'
import { useQueryClient } from '@tanstack/react-query'
import { db } from '@shared/db'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'
import type { SourceVersion } from '../data/types'


export interface WorkActions {
  // Work identity
  rename:        (title: string) => void
  setCover:      (url: string)   => void
  resetCover:    ()              => void

  // Library
  addLibrary:    ()                       => void
  removeLibrary: ()                       => void
  setStatus:     (status: LibraryStatus)  => void

  // Sources
  attachSource:  (s: WorkSource)                          => void
  detachSource:  (source: string, upstream_ref: string)   => void

  // Chapters
  /** Pack raw source pages into a BNL and save offline. */
  saveRawOffline:     (chapterRef: string, version: SourceVersion) => Promise<void>
}


const Ctx = createContext<WorkActions | null>(null)


export function useWorkActions(): WorkActions {
  const v = useContext(Ctx)
  if (!v) throw new Error('useWorkActions must be used inside <WorkActionsProvider>')
  return v
}


interface Props {
  workId:   string
  children: ReactNode
}


export function WorkActionsProvider({ workId, children }: Props) {
  const update    = useUpdateWork()
  const addLib    = useAddToLibrary()
  const rmLib     = useRemoveFromLibrary()
  const patchSt   = useUpdateLibraryStatus()
  const attach    = useAttachSource()
  const detach    = useDetachSource()
  const packRaw   = usePackRawArchive()

  const rename = useCallback((title: string) => {
    const t = title.trim()
    if (!t) return
    update.mutate({ id: workId, patch: { title: t, title_overridden: true } })
  }, [workId, update])

  const setCover = useCallback((url: string) => {
    update.mutate({ id: workId, patch: { cover_url: url, cover_overridden: true } })
  }, [workId, update])

  const resetCover = useCallback(() => {
    update.mutate({ id: workId, patch: { cover_overridden: false } })
  }, [workId, update])

  const addLibrary    = useCallback(() => addLib.mutate({ work_id: workId }), [workId, addLib])
  const removeLibrary = useCallback(() => rmLib.mutate(workId), [workId, rmLib])
  const setStatus     = useCallback(
    (s: LibraryStatus) => patchSt.mutate({ work_id: workId, status: s }),
    [workId, patchSt],
  )

  const attachSource = useCallback(
    (s: WorkSource) => attach.mutate({ work_id: workId, source: s }),
    [workId, attach],
  )
  const detachSource = useCallback(
    (source: string, upstream_ref: string) =>
      detach.mutate({ work_id: workId, source, upstream_ref }),
    [workId, detach],
  )

  const saveRawOffline = useCallback(
    async (chapterRef: string, version: SourceVersion) => {
      const pages = await fetchChapterPages(version.source.manifest, version.ref.url)
      if (!pages.pages.length) throw new Error('Chương không có trang.')
      await packRaw.mutateAsync({
        work_id:     workId,
        chapter_ref: chapterRef,
        raw_urls:    pages.pages,
      })
    },
    [workId, packRaw],
  )

  const value = useMemo<WorkActions>(() => ({
    rename, setCover, resetCover,
    addLibrary, removeLibrary, setStatus,
    attachSource, detachSource,
    saveRawOffline,
  }), [
    rename, setCover, resetCover,
    addLibrary, removeLibrary, setStatus,
    attachSource, detachSource,
    saveRawOffline,
  ])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}


// ── Internal helpers (mutations not yet exported elsewhere) ──────


/** Pack raw chapter pages into a BNL and persist to IDB. */
function usePackRawArchive() {
  const qc = useQueryClient()
  const { toBrowserUrl } = useSourceFetch()
  return useMutation({
    mutationFn: async (args: {
      work_id:     string
      chapter_ref: string
      raw_urls:    string[]
    }) => {
      const inputs: PackInput[] = await Promise.all(args.raw_urls.map(async (url) => {
        const res = await fetch(toBrowserUrl(url))
        if (!res.ok) throw new Error(`Trang lỗi: ${res.status}`)
        const blob = await res.blob()
        const data = await blob.arrayBuffer()
        const dims = await imageDimensions(blob)
        return {
          data,
          width:  dims.width,
          height: dims.height,
          format: detectFormat(blob.type),
        }
      }))
      const packed = await Bunle.pack(inputs)
      const blob   = new Blob([packed], { type: 'image/mcz' })
      const id     = `${args.work_id}:${args.chapter_ref}`
      await db().archives.put({
        id,
        work_id:     args.work_id,
        chapter_ref: args.chapter_ref,
        kind:        'raw',
        blob,
        page_count:  inputs.length,
        byte_size:   blob.size,
        saved_at:    new Date().toISOString(),
      })
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['archives'] }),
  })
}


function detectFormat(mime: string): 'webp' | 'jpeg' | 'jxl' {
  if (mime.includes('webp')) return 'webp'
  if (mime.includes('jxl'))  return 'jxl'
  return 'jpeg'
}

async function imageDimensions(blob: Blob): Promise<{ width: number; height: number }> {
  try {
    const bm = await createImageBitmap(blob)
    const out = { width: bm.width, height: bm.height }
    bm.close?.()
    return out
  } catch {
    return new Promise((resolve, reject) => {
      const img = new Image()
      const url = URL.createObjectURL(blob)
      img.onload  = () => {
        URL.revokeObjectURL(url)
        resolve({ width: img.naturalWidth, height: img.naturalHeight })
      }
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')) }
      img.src = url
    })
  }
}
