// useSpawnChapter — full pipeline: download pages → pack zip → upload
// → spawn translation.
//
// "Source picker theo chap user đã pick" means the caller passes the
// specific HubVersion (raw) the user wants to translate from. The hub
// already knows which source the user follows (primary material first
// in mergeChapters). The caller picks the version; this hook drives
// the pipeline.
//
// Progress phases:
//   fetching   — resolving page URL list from manifest
//   downloading — fetching image bytes (N/total pages)
//   packing    — building ZIP in memory
//   uploading  — multipart upload to R2 (0–100%)
//   spawning   — POST /api/translate
//   done / error

import { useState, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { useSources } from '@features/browse/sources'
import { packPagesToZip } from '@typoon/upload-sdk'
import { uploadChapterZip } from '@typoon/upload-sdk'
import type { HubVersion } from './mergeChapters'

export type SpawnPhase =
  | 'idle'
  | 'fetching'
  | 'downloading'
  | 'packing'
  | 'uploading'
  | 'spawning'
  | 'done'
  | 'error'

export interface SpawnProgress {
  phase:   SpawnPhase
  /** downloading: pages done so far */
  current: number
  /** downloading: total pages */
  total:   number
  /** uploading: 0–100 */
  pct:     number
  error:   string | null
}

const IDLE: SpawnProgress = {
  phase: 'idle', current: 0, total: 0, pct: 0, error: null,
}

export function useSpawnChapter(targetLang: string) {
  const [progress, setProgress] = useState<SpawnProgress>(IDLE)
  const sources = useSources((s) => s.sources)
  const qc      = useQueryClient()

  const spawn = useCallback(async (
    version: HubVersion,
    chapterLabel: string | null,
  ) => {
    if (!version.upstreamUrl || !version.sourceId || version.materialId == null) return
    const source = sources[version.sourceId]
    if (!source) {
      setProgress({ ...IDLE, phase: 'error', error: `Nguồn "${version.sourceId}" chưa được cài đặt.` })
      return
    }

    setProgress({ phase: 'fetching', current: 0, total: 0, pct: 0, error: null })

    try {
      // 1. Resolve page URLs from manifest.
      const { pages } = await fetchChapterPages(source.manifest, version.upstreamUrl)
      if (pages.length === 0) throw new Error('Nguồn không trả về trang nào.')

      // 2. Download pages — parallel with concurrency 4, fail fast.
      setProgress({ phase: 'downloading', current: 0, total: pages.length, pct: 0, error: null })

      const CONCURRENCY = 4
      let done = 0

      const queue = pages.map((url, i) => ({ url, i }))
      const results: { source: string; bytes: Uint8Array }[] = new Array(pages.length)

      const worker = async () => {
        while (queue.length > 0) {
          const item = queue.shift()
          if (!item) return
          const res = await fetch(item.url)
          if (!res.ok) throw new Error(`Tải trang ${item.i + 1} thất bại (HTTP ${res.status}).`)
          const buf = await res.arrayBuffer()
          results[item.i] = { source: item.url, bytes: new Uint8Array(buf) }
          done++
          setProgress((p) => ({ ...p, current: done }))
        }
      }

      await Promise.all(Array.from({ length: CONCURRENCY }, worker))

      // 3. Pack ZIP.
      setProgress({ phase: 'packing', current: pages.length, total: pages.length, pct: 0, error: null })
      const zip = packPagesToZip(results)

      // 4. Upload — server dedups by upstream_url so concurrent uploads
      //    of the same source chapter share one chapter row.
      setProgress({ phase: 'uploading', current: pages.length, total: pages.length, pct: 0, error: null })

      const chapter = await uploadChapterZip(api, version.materialId, zip, {
        label:       chapterLabel ?? undefined,
        upstreamUrl: version.upstreamUrl,
        numberNorm:  version.numberNorm ?? undefined,
        // Pin the chapter's source language to whatever the user
        // actually clicked. A MangaDex Italian material can host an
        // English-only chapter; without this, the server would
        // record `source_lang='it'` and the LLM gets told to
        // translate Italian text it can't see in the pixels.
        sourceLang:  version.lang || undefined,
        onProgress: (p) => {
          const pct = p.bytesTotal > 0 ? Math.round((p.bytesSent / p.bytesTotal) * 100) : 0
          setProgress((prev) => ({ ...prev, pct }))
        },
      })

      // 5. Spawn translation against the materialized chapter row.
      setProgress({ phase: 'spawning', current: pages.length, total: pages.length, pct: 100, error: null })

      await api.spawnTranslate({
        chapter_id:  chapter.id,
        target_lang: targetLang,
      })

      // Refresh work cache so the chapter list flips raw → running.
      // We don't know the specific workId in this scope; invalidate
      // every work entry (cheap at beta scale, one entry per open tab).
      await qc.invalidateQueries({ queryKey: qk.work.all() })

      setProgress({ phase: 'done', current: pages.length, total: pages.length, pct: 100, error: null })
    } catch (err) {
      setProgress((p) => ({
        ...p,
        phase: 'error',
        error: err instanceof Error ? err.message : 'Lỗi không xác định.',
      }))
    }
  }, [sources, targetLang, qc])

  const reset = useCallback(() => setProgress(IDLE), [])

  return { progress, spawn, reset }
}
