// Offline archives — Dexie store of saved raw .bnl files.
//
// Stored blobs are valid BNL files the reader can `Bunle.from(buf)`
// directly. Translation is rendered live in the reader overlay.

import { useLiveQuery } from 'dexie-react-hooks'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Bunle, type PackInput } from '@nghyane/bunle'

import { db, type SavedArchive } from '@shared/db'
import { useSourceFetch } from '@features/browse/SourceFetchProvider'

export type { SavedArchive }

function archiveId(work_id: string, chapter_ref: string): string {
  return `${work_id}:${chapter_ref}`
}


// ── Lookup ──────────────────────────────────────────────────────────

/** Live archive presence — re-renders when the user saves/deletes. */
export function useSavedArchive(
  work_id:     string | null | undefined,
  chapter_ref: string | null | undefined,
): SavedArchive | null {
  return useLiveQuery(async () => {
    if (!work_id || !chapter_ref) return null
    return (await db().archives.get(archiveId(work_id, chapter_ref))) ?? null
  }, [work_id, chapter_ref]) ?? null
}


// ── Pack raw chapter → BNL ──────────────────────────────────────────

/** Pack a list of raw upstream image URLs into a BNL file and save it
 *  to IndexedDB. Useful for offline reading of un-translated chapters. */
export function usePackRawArchive() {
  const qc = useQueryClient()
  const { toBrowserUrl } = useSourceFetch()
  return useMutation({
    mutationFn: async (args: {
      work_id:     string
      chapter_ref: string
      raw_urls:    string[]
    }) => {
      const inputs = await Promise.all(args.raw_urls.map(async (url) => {
        const proxied = toBrowserUrl(url)
        const res = await fetch(proxied)
        if (!res.ok) throw new Error(`Raw page fetch failed: ${res.status} ${url}`)
        const blob = await res.blob()
        const data = await blob.arrayBuffer()
        const dims = await imageDimensions(blob)
        return {
          data,
          width:  dims.width,
          height: dims.height,
          format: detectFormat(blob.type),
        } satisfies PackInput
      }))

      const packed = await Bunle.pack(inputs)
      const blob   = new Blob([packed], { type: 'image/mcz' })
      const item: SavedArchive = {
        id:          archiveId(args.work_id, args.chapter_ref),
        work_id:     args.work_id,
        chapter_ref: args.chapter_ref,
        kind:        'raw',
        blob,
        page_count:  inputs.length,
        byte_size:   blob.size,
        saved_at:    new Date().toISOString(),
      }
      await db().archives.put(item)
      return item
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['archives'] }),
  })
}


// ── Delete ──────────────────────────────────────────────────────────

export function useDeleteArchive() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: { work_id: string; chapter_ref: string }) => {
      await db().archives.delete(archiveId(args.work_id, args.chapter_ref))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['archives'] }),
  })
}


// ── Storage telemetry ───────────────────────────────────────────────

export function useArchiveStorageStats() {
  return useLiveQuery(async () => {
    const all = await db().archives.toArray()
    const total_bytes = all.reduce((s, a) => s + a.byte_size, 0)
    return {
      count:        all.length,
      total_bytes,
      raw:          all.filter(a => a.kind === 'raw').length,
    }
  }, [])
}


// ── Helpers ─────────────────────────────────────────────────────────

function detectFormat(mime: string): 'webp' | 'jpeg' | 'jxl' {
  if (mime.includes('webp')) return 'webp'
  if (mime.includes('jxl'))  return 'jxl'
  return 'jpeg'             // default for jpg, png-as-jpg fallbacks
}

async function imageDimensions(blob: Blob): Promise<{ width: number; height: number }> {
  // ImageBitmap is the fastest path; falls back to Image element if a
  // codec (e.g. JXL) isn't supported by the browser's bitmap decoder.
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
        const out = { width: img.naturalWidth, height: img.naturalHeight }
        URL.revokeObjectURL(url)
        resolve(out)
      }
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Image decode failed')) }
      img.src = url
    })
  }
}
