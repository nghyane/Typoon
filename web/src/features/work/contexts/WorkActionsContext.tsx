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
import {
  useDownloadTranslatedArchive,
} from '@features/reader/archives'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import { useSubmitJob } from '@features/jobs/useSubmitJob'
import { packPagesToZip } from '@typoon/upload-sdk'
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
  /** Spawn (or re-spawn) a translate job. Same chapter_ref creates a
   *  new job; older job rows stay in IDB for the row's history. */
  spawnTranslate:     (chapterRef: string, version: SourceVersion) => Promise<void>
  /** Pack raw source pages into a BNL and save offline. */
  saveRawOffline:     (chapterRef: string, version: SourceVersion) => Promise<void>
  /** Download a server-translated archive and save offline. */
  downloadTranslated: (chapterRef: string, jobId: number, archiveUrl: string) => Promise<void>
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
  const dlBnl     = useDownloadTranslatedArchive()
  const submitJob = useSubmitJob()
  const packRaw   = usePackRawArchive()
  const spawnHelp = useSpawnTranslateHelper(workId, submitJob.submit)

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

  const spawnTranslate = useCallback(
    (chapterRef: string, version: SourceVersion) => spawnHelp(chapterRef, version),
    [spawnHelp],
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

  const downloadTranslated = useCallback(
    async (chapterRef: string, jobId: number, archiveUrl: string) => {
      await dlBnl.mutateAsync({
        work_id:     workId,
        chapter_ref: chapterRef,
        job_id:      jobId,
        archive_url: archiveUrl,
      })
    },
    [workId, dlBnl],
  )

  const value = useMemo<WorkActions>(() => ({
    rename, setCover, resetCover,
    addLibrary, removeLibrary, setStatus,
    attachSource, detachSource,
    spawnTranslate, saveRawOffline, downloadTranslated,
  }), [
    rename, setCover, resetCover,
    addLibrary, removeLibrary, setStatus,
    attachSource, detachSource,
    spawnTranslate, saveRawOffline, downloadTranslated,
  ])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}


// ── Internal helpers (mutations not yet exported elsewhere) ──────


/** Pack raw chapter pages into a BNL and persist to IDB. Mirrors the
 *  legacy `usePackRawArchive` but stays internal to this module so the
 *  action context owns its mutations. */
function usePackRawArchive() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      work_id:     string
      chapter_ref: string
      raw_urls:    string[]
    }) => {
      const inputs: PackInput[] = await Promise.all(args.raw_urls.map(async (url) => {
        const res = await fetch(proxify(url))
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


/** Spawn translate job — fetch source pages, zip them, submit to /jobs.
 *  Bound to `workId` so callers only pass chapter-level info. */
function useSpawnTranslateHelper(
  workId: string,
  submit: ReturnType<typeof useSubmitJob>['submit'],
) {
  return useCallback(async (chapterRef: string, version: SourceVersion) => {
    const pages = await fetchChapterPages(version.source.manifest, version.ref.url)
    if (!pages.pages.length) throw new Error('Chương trống.')
    const bytes = await Promise.all(pages.pages.map(async (url, i) => {
      const res = await fetch(proxify(url))
      if (!res.ok) throw new Error(`Trang ${i + 1} lỗi ${res.status}`)
      return { source: url, bytes: new Uint8Array(await res.arrayBuffer()) }
    }))
    const zip = packPagesToZip(bytes)
    await submit({
      work_id:     workId,
      chapter_ref: chapterRef,
      source_lang: version.ref.language
                ?? version.source.manifest.languages[0]
                ?? 'ja',
      kind:        'translate',
      zip,
    })
  }, [workId, submit])
}
