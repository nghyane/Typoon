// Translation orchestration — drives the full client→R2→pipeline flow.
//
//   1. POST /jobs (with work_id, byte_size, kind)
//   2. PUT each presigned part to R2 in parallel
//   3. POST /jobs/:id/start
//   4. Caller polls via useJob() until state='done' | 'error'
//
// Context is server-side: when work_id is supplied, the server pulls
// the latest KV blob into ctx/{job_id}/input.json.gz before pipeline
// kicks off. After finalize, the server writes the merged context back
// to KV — the client just refetches `qk.context.byWork(workId)` to see
// updates.

import { useCallback, useState } from 'react'

import { api, type ApiJob, type JobKind } from '@shared/api/api'
import { db } from '@shared/db'

const CONCURRENCY    = 4
const PUT_ATTEMPTS   = 3
const RETRY_BACKOFFS = [250, 1000, 4000] as const

export interface TranslateChapterInput {
  /** Composite work id (e.g. "mdex:abc-uuid"). When provided, the
   *  server hydrates context from KV before the pipeline runs. */
  work_id?:     string
  /** Optional chapter ref for IndexedDB mirror — used by the reader to
   *  resolve "translated archive for this chapter". */
  chapter_ref?: string
  source_lang:  string
  target_lang?: string
  kind?:        JobKind        // default 'translate'
  zip:          Blob
  onProgress?:  (loaded: number, total: number) => void
  signal?:      AbortSignal
}

export interface TranslateChapterResult {
  job_id: number
  /** Brief snapshot returned by /start (post-state transition). */
  job:    ApiJob
}

class PermanentPutError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}


// ── Public API ───────────────────────────────────────────────────────

export function useSubmitJob() {
  const [progress, setProgress] = useState({ loaded: 0, total: 0 })

  const submit = useCallback(async (
    input: TranslateChapterInput,
  ): Promise<TranslateChapterResult> => {
    const total = input.zip.size
    setProgress({ loaded: 0, total })

    // 1. init
    const init = await api.jobsCreate({
      byte_size:   total,
      source_lang: input.source_lang,
      target_lang: input.target_lang,
      work_id:     input.work_id,
      kind:        input.kind ?? 'translate',
    })

    // Mirror into IDB so the reader can find the job by chapter_ref
    // even before it reaches 'done'.
    await db().jobs.put({
      id:              init.job_id,
      work_id:         input.work_id ?? null,
      chapter_ref:     input.chapter_ref ?? null,
      kind:            init.kind,
      state:           'uploading',
      archive_url:     null,
      archive_expires: null,
      page_count:      null,
      created_at:      new Date().toISOString(),
      expires_at:      new Date(Date.now() + 7 * 86_400_000).toISOString(),
    })

    // 2. upload parts in parallel
    const completed = await uploadParts({
      parts:     init.parts,
      part_size: init.part_size,
      zip:       input.zip,
      onProgress: (loaded) => {
        setProgress({ loaded, total })
        input.onProgress?.(loaded, total)
      },
      signal: input.signal,
    })

    // 3. start
    const job = await api.jobsStart(init.job_id, completed)

    // Sync IDB state forward
    await db().jobs.update(init.job_id, { state: job.state })

    return { job_id: init.job_id, job }
  }, [])

  return { submit, progress }
}


// ── Multipart upload driver ──────────────────────────────────────────

interface UploadPartsInput {
  parts:     { number: number; url: string }[]
  part_size: number
  zip:       Blob
  onProgress?: (loaded: number) => void
  signal?:   AbortSignal
}

async function uploadParts(input: UploadPartsInput): Promise<{
  number: number; etag: string
}[]> {
  if (input.parts.length === 0) {
    throw new Error('upload-init returned zero parts')
  }

  const counters = new Map<number, number>()
  const total    = input.zip.size

  const reportTotal = () => {
    let sum = 0
    for (const v of counters.values()) sum += v
    input.onProgress?.(Math.min(sum, total))
  }

  const queue: typeof input.parts = [...input.parts]
  const completed: { number: number; etag: string }[] = []
  let failed: unknown = null

  const worker = async () => {
    while (queue.length > 0 && failed === null) {
      const part = queue.shift()
      if (!part) return
      const start = (part.number - 1) * input.part_size
      const end   = Math.min(start + input.part_size, total)
      const slice = input.zip.slice(start, end)

      try {
        const etag = await putWithRetry(part.number, part.url, slice, {
          onProgress: (loaded) => {
            counters.set(part.number, loaded)
            reportTotal()
          },
          signal: input.signal,
        })
        counters.set(part.number, slice.size)
        reportTotal()
        completed.push({ number: part.number, etag })
      } catch (e) {
        failed = e
        return
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, input.parts.length) }, () => worker()),
  )

  if (failed) {
    if (failed instanceof PermanentPutError) {
      throw new Error(`PUT ${failed.status}: ${failed.message}`)
    }
    throw failed
  }
  if (completed.length !== input.parts.length) {
    throw new Error(
      `upload incomplete: ${completed.length}/${input.parts.length} parts`,
    )
  }
  completed.sort((a, b) => a.number - b.number)
  return completed
}


// ── Per-part PUT with retry ─────────────────────────────────────────

async function putWithRetry(
  partNumber: number,
  url:        string,
  body:       Blob,
  opts:       {
    onProgress?: (loaded: number) => void
    signal?:     AbortSignal
  },
): Promise<string> {
  let lastErr: unknown
  for (let attempt = 0; attempt < PUT_ATTEMPTS; attempt++) {
    if (opts.signal?.aborted) throw new DOMException('aborted', 'AbortError')
    if (attempt > 0) {
      await sleep(RETRY_BACKOFFS[attempt - 1] ?? 4000, opts.signal)
    }
    try {
      return await putOnce(url, body, opts)
    } catch (e) {
      lastErr = e
      if (e instanceof PermanentPutError) throw e
      if ((e as { name?: string })?.name === 'AbortError') throw e
    }
  }
  throw lastErr ?? new Error(`PUT part ${partNumber} failed`)
}

function putOnce(
  url:  string,
  body: Blob,
  opts: {
    onProgress?: (loaded: number) => void
    signal?:     AbortSignal
  },
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url, true)

    xhr.upload.onprogress = (e) => {
      if (opts.onProgress && e.lengthComputable) opts.onProgress(e.loaded)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const raw = xhr.getResponseHeader('ETag')
        if (!raw) {
          reject(new PermanentPutError(
            xhr.status,
            'Missing ETag — kiểm tra CORS bucket (ExposeHeaders: ETag).',
          ))
          return
        }
        resolve(raw.replace(/^"|"$/g, ''))
      } else if (xhr.status >= 400 && xhr.status < 500) {
        reject(new PermanentPutError(xhr.status, `PUT ${xhr.status}`))
      } else {
        reject(new Error(`PUT ${xhr.status}`))
      }
    }
    xhr.onerror   = () => reject(new Error('network'))
    xhr.ontimeout = () => reject(new Error('timeout'))

    if (opts.signal) {
      if (opts.signal.aborted) {
        xhr.abort()
        reject(new DOMException('aborted', 'AbortError'))
        return
      }
      opts.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }

    xhr.send(body)
  })
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    if (signal) {
      if (signal.aborted) {
        clearTimeout(t)
        reject(new DOMException('aborted', 'AbortError'))
        return
      }
      signal.addEventListener('abort', () => {
        clearTimeout(t)
        reject(new DOMException('aborted', 'AbortError'))
      }, { once: true })
    }
  })
}
