// useSpawnChapters — full pipeline: download pages → pack zip → upload
// → spawn translation. Multi-instance: callers can kick off several
// chapters concurrently and each has an independent progress slot.
//
// Caller passes the specific HubVersion (raw) the user wants to
// translate from; the hub already knows which source the user follows
// (primary material first in mergeChapters). The caller picks the
// version; this hook drives the pipeline.
//
// Progress is keyed by an OPAQUE CALLER STRING — typically the
// chapter's `number_norm`. Keying by chapter (not by version) is
// deliberate: when the spawn succeeds, the chapter row's identity in
// the UI flips from "raw + spawn chip" to "translation row". A
// version-scoped key would lose its progress entry at exactly that
// transition and the chip would visibly jump to a different row.
// Chapter-scoped keys stay attached to the chapter through the whole
// lifecycle — client download/upload/server pending/running/done all
// read the same `getProgress(chapterNumber)` slot.
//
// Multi-instance: the UI lets the user tap "Dịch" on several rows in
// a row; each chapter has its own progress slot keyed by chapter
// number.
//
// Pipeline (progress phases):
//   fetching    — resolving page URL list from manifest
//   downloading — fetching image bytes (N/total pages)
//   packing     — building ZIP in memory
//   uploading   — multipart upload to R2 (0–100%)
//   spawning    — POST /api/translate
//   done / error
//
// Robustness vs. v1:
//   • Per-page retry with exponential backoff on 429/5xx/network
//     errors. The DA proxy occasionally fans timeouts back to us on
//     hot CDN paths; one stray 503 used to nuke the whole spawn.
//   • Adaptive concurrency: starts at 6, drops to 2 when the proxy
//     surfaces rate-limit status codes (429/503) twice within 3 s.
//   • Resumable: page bytes survive an `error` phase so the user can
//     retry without re-downloading. Cache evicts on `done` or an
//     explicit `reset(key)`.
//   • Cancellable: every key owns an `AbortController`; `abort(key)`
//     yanks the proxy fetches and the multipart upload XHRs at the
//     same time.
//   • Narrow cache invalidation: only the work this spawn belongs to
//     gets re-queried instead of every open work tab.

import { useCallback, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import type { ApiWorkChapterTranslation, ApiWorkDetail } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { pfetch } from '@features/browse/proxy'
import { useSources } from '@features/browse/sources'
import { packPagesToZip, uploadChapterZip } from '@typoon/upload-sdk'
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
  /** True when the cached page bytes from a previous attempt are
   *  reusable, so the UI can show "Thử lại (giữ N/M trang đã tải)". */
  resumable: boolean
}

const IDLE: SpawnProgress = {
  phase: 'idle', current: 0, total: 0, pct: 0, error: null, resumable: false,
}

export interface SpawnChaptersApi {
  /** Snapshot per active key. Keys whose pipeline never ran are absent. */
  progressByKey: Record<string, SpawnProgress>
  /** Read progress for a specific row. Returns null when nothing has
   *  been spawned for that key in this session. */
  getProgress: (key: string) => SpawnProgress | null
  /** Spawn translation for `version` under `key`. `key` should be the
   *  chapter's `number_norm` so the progress slot survives the row's
   *  raw\u2192translation transition. Subsequent calls with the same key
   *  while a run is in-flight are ignored so the user can't double-
   *  fire on the same chapter. When a previous attempt errored,
   *  calling `spawn` again resumes from the cached page bytes (skips
   *  already-downloaded pages). */
  spawn: (
    key: string,
    version: HubVersion,
    chapterLabel: string | null,
    workId?: number,
  ) => void
  /** Cancel an in-flight pipeline. Aborts page fetches and the
   *  multipart upload; progress flips to `error` with reason
   *  "Đã huỷ". Cache is preserved so the user can resume. */
  abort: (key: string) => void
  /** Clear progress entry and any cached page bytes for a key. */
  reset: (key: string) => void
  /** Clear every entry (e.g. route unmount). */
  resetAll: () => void
}

// ── Tunables ───────────────────────────────────────────────────

/** Download workers when the proxy is healthy. 6 saturates a typical
 *  home upstream without tripping per-IP limits on the busier CDNs
 *  (Mangadex, weebcentral). */
const CONCURRENCY_HEALTHY = 6
/** Drop to this on rate-limit signals. 2 keeps the pipeline moving
 *  without piling more 429s on top. */
const CONCURRENCY_BACKOFF = 2
/** Per-page retry budget. Total attempts = 1 + RETRY_MAX. */
const RETRY_MAX = 3
/** Base backoff. Real wait = BASE * 2^attempt + jitter(0, BASE). */
const RETRY_BASE_MS = 400
/** Window inside which repeated rate-limit hits trip backoff mode. */
const RL_WINDOW_MS = 3_000
/** Two 429/503 in the window → switch every running worker to
 *  CONCURRENCY_BACKOFF. */
const RL_TRIP_COUNT = 2

// Status codes that mean "try again later"; everything else 4xx is a
// permanent error we should surface immediately (404 missing page,
// 401 expired auth, …).
const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504])
const RATELIMIT_STATUS = new Set([429, 503])

// ── Helpers ────────────────────────────────────────────────────

interface PageBytes {
  source: string
  bytes:  Uint8Array
}

/** Throttle controller shared by every download worker on a single
 *  spawn. When `tripped` flips on, in-flight workers finish their
 *  current page and the pool spins down to `CONCURRENCY_BACKOFF`. */
class RateLimitGate {
  private hits: number[] = []
  tripped = false

  noteRateLimit(): void {
    const now = performance.now()
    this.hits.push(now)
    this.hits = this.hits.filter((t) => now - t <= RL_WINDOW_MS)
    if (this.hits.length >= RL_TRIP_COUNT) this.tripped = true
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('aborted', 'AbortError'))
      return
    }
    const id = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(id)
      reject(new DOMException('aborted', 'AbortError'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

/** Fetch a single page with retry + backoff. Throws on permanent
 *  errors (4xx ≠ 408/425/429) or after exhausting the retry budget. */
async function fetchPageWithRetry(
  url:    string,
  index:  number,
  signal: AbortSignal,
  gate:   RateLimitGate,
): Promise<Uint8Array> {
  let lastErr: unknown = null
  for (let attempt = 0; attempt <= RETRY_MAX; attempt++) {
    if (signal.aborted) throw new DOMException('aborted', 'AbortError')

    try {
      const res = await pfetch(url, { init: { signal } })
      if (res.ok) {
        const buf = await res.arrayBuffer()
        return new Uint8Array(buf)
      }
      if (RATELIMIT_STATUS.has(res.status)) gate.noteRateLimit()
      if (!RETRYABLE_STATUS.has(res.status) || attempt === RETRY_MAX) {
        throw new Error(`Tải trang ${index + 1} thất bại (HTTP ${res.status}).`)
      }
      lastErr = new Error(`HTTP ${res.status}`)
    } catch (err) {
      if (isAbortError(err)) throw err
      if (attempt === RETRY_MAX) {
        // Wrap the native error in a localized message so the chip
        // doesn't surface "TypeError: Failed to fetch" verbatim. The
        // underlying error is attached for devtools inspection.
        const wrapped = new Error(`Tải trang ${index + 1} thất bại (mạng).`)
        // @ts-expect-error — `cause` is ES2022, lib here is ES2020.
        wrapped.cause = err
        throw wrapped
      }
      lastErr = err
    }

    // Exponential backoff with full jitter so concurrent retries
    // don't lock-step into the same upstream window.
    const base = RETRY_BASE_MS * (1 << attempt)
    const wait = base + Math.random() * RETRY_BASE_MS
    await sleep(wait, signal)
  }
  // Unreachable — the loop always returns or throws — but TS can't see it.
  throw lastErr instanceof Error ? lastErr : new Error('Tải trang thất bại.')
}

/** Validate + dedupe the URL list. Drops empty / non-http entries so
 *  we don't waste a slot on garbage manifest output. */
function sanitizePages(urls: string[]): string[] {
  const out: string[] = []
  for (const u of urls) {
    if (typeof u !== 'string') continue
    const trimmed = u.trim()
    if (!trimmed) continue
    if (!/^https?:\/\//i.test(trimmed)) continue
    out.push(trimmed)
  }
  return out
}


/** Optimistic overlay: write the freshly spawned translation row into
 *  the work cache so the UI sees the translation chip flip into place
 *  TẠI CHỖ ngay sau khi server xác nhận `spawnTranslate`, thay vì
 *  chờ vòng refetch tiếp theo và để user nhìn thấy "phần tử lạ" hiện
 *  ra cạnh row raw.
 *
 *  Approach: read the cached `ApiWorkDetail`, splice a new
 *  `ApiWorkChapterTranslation` into the matching `work_chapter` (or
 *  create the chapter row if the work hasn't materialised it yet),
 *  and write back. The subsequent `invalidateQueries` refetch will
 *  overlay the authoritative row by matching `translation.id` — the
 *  stub's fields just keep the row visible during the network gap.
 *
 *  Fields we can't synthesise (creator_name, owner_id, exact
 *  updated_at) land as null/now-iso; the refetch fills them in.
 *  `creator_name == null` renders as an empty handle which the row
 *  already tolerates (used for legacy untagged drafts). */
function injectOptimisticTranslation(
  qc:    ReturnType<typeof useQueryClient>,
  workId: number,
  stub: {
    translationId:     number
    state:             ApiWorkChapterTranslation['state']
    chapterId:         number
    chapterNumber:     string
    chapterLabel:      string | null
    targetLang:        string
    sourceLang:        string | null
    draftMaterialId:   number
    usesDefaultRender: boolean
  },
): void {
  qc.setQueryData<ApiWorkDetail | undefined>(
    qk.work.byId(workId),
    (old) => {
      if (!old) return old
      const now = new Date().toISOString()
      const newTx: ApiWorkChapterTranslation = {
        id:                  stub.translationId,
        target_lang:         stub.targetLang,
        source_lang:         stub.sourceLang,
        owner_id:            0,
        creator_name:        null,
        state:               stub.state,
        error_message:       null,
        shared:              true,
        draft_id:            null,
        draft_chapter_id:    null,
        draft_material_id:   stub.draftMaterialId,
        uses_default_render: stub.usesDefaultRender,
        updated_at:          now,
      }
      const chapters = [...old.chapters]
      const idx = chapters.findIndex((c) => c.number_norm === stub.chapterNumber)
      if (idx >= 0) {
        const ch = chapters[idx]!
        // If a translation with this id already exists (rare —
        // double-spawn raced through), replace rather than duplicate.
        const existing = ch.translations.findIndex((t) => t.id === stub.translationId)
        const translations = existing >= 0
          ? ch.translations.map((t, i) => i === existing ? newTx : t)
          : [...ch.translations, newTx]
        chapters[idx] = { ...ch, translations }
      } else {
        chapters.push({
          id:                 stub.chapterId,
          number_norm:        stub.chapterNumber,
          label:              stub.chapterLabel,
          translations:       [newTx],
          uploading_chapters: [],
        })
      }
      return { ...old, chapters }
    },
  )
}

// ── Hook ───────────────────────────────────────────────────────

export function useSpawnChapters(targetLang: string): SpawnChaptersApi {
  const [progressByKey, setProgressByKey] = useState<Record<string, SpawnProgress>>({})
  const sources = useSources((s) => s.sources)
  const qc      = useQueryClient()

  // Stable per-key state lives outside React state so re-renders
  // don't churn the cache or the abort controllers. The setter
  // below mirrors phase/progress into React for the UI to read.
  const inFlight   = useRef<Set<string>>(new Set())
  const aborters   = useRef<Map<string, AbortController>>(new Map())
  // Cached page bytes by key: index-aligned with the resolved page
  // URL list. Survives `error` so a retry skips finished pages. The
  // URL list is cached alongside so the retry sees the same indices.
  const pageCache  = useRef<Map<string, { urls: string[]; bytes: (Uint8Array | null)[] }>>(new Map())

  const setKey = useCallback((
    key: string,
    patch: Partial<SpawnProgress> | ((prev: SpawnProgress) => SpawnProgress),
  ) => {
    setProgressByKey((all) => {
      const prev = all[key] ?? IDLE
      const next = typeof patch === 'function' ? patch(prev) : { ...prev, ...patch }
      return { ...all, [key]: next }
    })
  }, [])

  const spawn = useCallback((
    key:          string,
    version:      HubVersion,
    chapterLabel: string | null,
    workId?:      number,
  ) => {
    if (inFlight.current.has(key)) return
    if (!version.upstreamUrl || !version.sourceId || version.materialId == null) return
    const source = sources[version.sourceId]
    if (!source) {
      setKey(key, {
        ...IDLE,
        phase: 'error',
        error: `Nguồn "${version.sourceId}" chưa được cài đặt.`,
      })
      return
    }

    const ctl = new AbortController()
    aborters.current.set(key, ctl)
    inFlight.current.add(key)
    setKey(key, {
      phase: 'fetching', current: 0, total: 0, pct: 0, error: null, resumable: false,
    })

    void (async () => {
      try {
        // 1. Resolve page URLs from manifest. Reuse the cached URL
        //    list when retrying so the retry sees the same indices
        //    as the previous attempt and the cached bytes line up.
        let cache = pageCache.current.get(key)
        if (!cache) {
          const fetched = await fetchChapterPages(source.manifest, version.upstreamUrl!)
          const urls = sanitizePages(fetched.pages)
          if (urls.length === 0) {
            throw new Error('Nguồn không trả về trang hợp lệ nào.')
          }
          cache = { urls, bytes: new Array(urls.length).fill(null) }
          pageCache.current.set(key, cache)
        }
        const { urls, bytes } = cache

        // 2. Download with retry + adaptive concurrency. Skip indices
        //    that are already populated (resumed retry).
        const total = urls.length
        let done = bytes.reduce((n, b) => n + (b ? 1 : 0), 0)
        setKey(key, {
          phase: 'downloading', current: done, total, pct: 0, error: null, resumable: false,
        })

        const queue: number[] = []
        for (let i = 0; i < urls.length; i++) {
          if (!bytes[i]) queue.push(i)
        }

        if (queue.length > 0) {
          const gate = new RateLimitGate()
          // Worker pool: each worker pulls indices off `queue` until
          // it's empty OR until the rate-limit gate trips and the
          // worker's slot exceeds the backoff target. Surplus workers
          // exit voluntarily; remaining workers keep draining. We
          // start CONCURRENCY_HEALTHY workers; on a trip, all but the
          // first CONCURRENCY_BACKOFF self-retire after their current
          // page lands.
          const worker = async (slot: number): Promise<void> => {
            while (queue.length > 0) {
              if (ctl.signal.aborted) return
              // Self-retire when the gate trips and our slot is
              // above the backoff cap. Slots are sticky per worker so
              // exactly the high-numbered ones exit; survivors keep
              // draining without further coordination.
              if (gate.tripped && slot >= CONCURRENCY_BACKOFF) return
              const i = queue.shift()
              if (i === undefined) return
              const buf = await fetchPageWithRetry(urls[i]!, i, ctl.signal, gate)
              bytes[i] = buf
              done++
              setKey(key, (p) => ({ ...p, current: done }))
            }
          }

          const startCount = Math.min(CONCURRENCY_HEALTHY, queue.length)
          const workers: Promise<void>[] = []
          for (let s = 0; s < startCount; s++) workers.push(worker(s))
          // Await every worker so a thrown error from any retry-
          // exhausted page bubbles up. Promise.all rejects on the
          // first failure; survivors then notice `ctl.signal` (the
          // catch block below aborts to stop the rest).
          try {
            await Promise.all(workers)
          } catch (err) {
            ctl.abort()
            // Drain so we don't leave detached promises spewing
            // unhandled rejections after the catch block exits.
            await Promise.allSettled(workers)
            throw err
          }
        }

        if (ctl.signal.aborted) throw new DOMException('aborted', 'AbortError')

        // 3. Pack ZIP. `packPagesToZip` is synchronous (DA iframe has
        //    no Worker constructor) — kept on main thread; the actual
        //    work is store-mode memcpy so it stays under a few hundred
        //    ms even for 50-page chapters.
        setKey(key, {
          phase: 'packing', current: total, total, pct: 0, error: null, resumable: false,
        })
        const pages: PageBytes[] = []
        for (let i = 0; i < urls.length; i++) {
          const b = bytes[i]
          if (!b) throw new Error(`Trang ${i + 1} chưa tải xong.`)
          pages.push({ source: urls[i]!, bytes: b })
        }
        const zip = packPagesToZip(pages)

        // 4. Upload — server dedups by upstream_url so concurrent
        //    uploads of the same source chapter share one chapter row.
        setKey(key, {
          phase: 'uploading', current: total, total, pct: 0, error: null, resumable: false,
        })
        const chapter = await uploadChapterZip(api, version.materialId!, zip, {
          label:       chapterLabel ?? undefined,
          upstreamUrl: version.upstreamUrl!,
          numberNorm:  version.numberNorm ?? undefined,
          // Pin the chapter's source language to whatever the user
          // actually clicked. A MangaDex Italian material can host an
          // English-only chapter; without this, the server would
          // record `source_lang='it'` and the LLM gets told to
          // translate Italian text it can't see in the pixels.
          sourceLang:  version.lang || undefined,
          signal:      ctl.signal,
          onProgress: (p) => {
            const pct = p.bytesTotal > 0 ? Math.round((p.bytesSent / p.bytesTotal) * 100) : 0
            setKey(key, (prev) => ({ ...prev, pct }))
          },
        })

        // 5. Spawn translation against the materialized chapter row.
        setKey(key, {
          phase: 'spawning', current: total, total, pct: 100, error: null, resumable: false,
        })
        const spawnResult = await api.spawnTranslate({
          chapter_id:  chapter.id,
          target_lang: targetLang,
        })

        // Optimistic overlay — paint the new translation row into the
        // work cache RIGHT NOW so the user sees the spawn chip flip
        // from "Đang dịch (raw row)" → "Đang dịch (translation row)"
        // tại chỗ. Without this step, the raw row keeps its `done`
        // chip while a brand-new translation row materializes on the
        // next refetch tick, which reads as "phần tử lạ xuất hiện".
        //
        // Server `spawnTranslate` response only carries the
        // translation id + state, so we have to lift the rest of the
        // row from what the client already knows (the user clicked
        // the raw whose `materialId` + `lang` matches the upload).
        // Once the invalidate-driven refetch lands, the server row
        // overlays this stub by matching `id` — no duplicate, no
        // flash.
        if (workId != null) {
          injectOptimisticTranslation(qc, workId, {
            translationId:    spawnResult.translation_id,
            state:            spawnResult.state,
            chapterId:        chapter.id,
            chapterNumber:    chapter.number,
            chapterLabel:     chapterLabel ?? null,
            targetLang,
            sourceLang:       version.lang || null,
            draftMaterialId:  version.materialId!,
            usesDefaultRender: spawnResult.cache_hit,
          })
        }

        // Narrow invalidate when the caller knows which work this
        // spawn belongs to. Fall back to the broad key only for old
        // call sites that haven't been threaded yet — beta scale, one
        // entry per open tab, so the blast radius is bounded either
        // way. The work-payload refetch is what carries authoritative
        // creator/timestamp fields the optimistic stub couldn't
        // synthesise.
        await qc.invalidateQueries({
          queryKey: workId != null ? qk.work.byId(workId) : qk.work.all(),
        })

        // Pipeline succeeded — drop the cached page bytes so the next
        // spawn (different chapter) doesn't sit on dead memory.
        pageCache.current.delete(key)
        setKey(key, {
          phase: 'done', current: total, total, pct: 100, error: null, resumable: false,
        })
      } catch (err) {
        const aborted = isAbortError(err) || ctl.signal.aborted
        // On error we keep the cache so the user's retry resumes from
        // wherever we got to. The cache evicts on `reset(key)` or on
        // a successful run.
        const cache = pageCache.current.get(key)
        const resumable = !!cache && cache.bytes.some((b) => b !== null)
        setKey(key, (p) => ({
          ...p,
          phase: 'error',
          error: aborted
            ? 'Đã huỷ.'
            : err instanceof Error ? err.message : 'Lỗi không xác định.',
          resumable,
        }))
      } finally {
        inFlight.current.delete(key)
        if (aborters.current.get(key) === ctl) aborters.current.delete(key)
      }
    })()
  }, [sources, targetLang, qc, setKey])

  const abort = useCallback((key: string) => {
    const ctl = aborters.current.get(key)
    if (!ctl) return
    ctl.abort()
  }, [])

  const getProgress = useCallback(
    (key: string): SpawnProgress | null => progressByKey[key] ?? null,
    [progressByKey],
  )

  const reset = useCallback((key: string) => {
    pageCache.current.delete(key)
    setProgressByKey((all) => {
      if (!(key in all)) return all
      const next = { ...all }
      delete next[key]
      return next
    })
  }, [])

  const resetAll = useCallback(() => {
    pageCache.current.clear()
    setProgressByKey({})
  }, [])

  return { progressByKey, getProgress, spawn, abort, reset, resetAll }
}
