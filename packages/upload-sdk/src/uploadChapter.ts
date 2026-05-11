// Top-level upload driver: pack → init → put parts in parallel → finalize.
//
// Flow matches the engine routes in `typoon/api/routes/upload.py`:
//
//   1. POST /chapters/upload-init  → presigned PUTs
//   2. PUT each part to its presigned URL (4 in flight by default)
//   3. POST /chapters/upload-finalize { parts: [{number, etag}], ... }
//
// Cancellation: an `AbortSignal` aborts in-flight XHRs and triggers a
// best-effort `/chapters/upload-abort` so the inbox key doesn't sit
// around until the bucket lifecycle rule sweeps it.
//
// Progress: per-byte tracked across parts (see `ProgressTracker`),
// debounced 250ms.

import { ProgressTracker, type ProgressCallback } from './progress'
import { PermanentPutError, putPart, type PutPartResult } from './putPart'
import type {
  ApiChapterLike, FinalizePart, UploadHttpClient,
} from './api'

const DEFAULT_CONCURRENCY = 4

export interface UploadOptions {
  number?:      string
  /** Free-form chapter label (e.g. "Extra: Volume 1 Cover"). */
  label?:       string
  /** How many part PUTs to run in parallel. Default 4 — saturates a
   *  typical home upstream while staying inside R2 free-tier
   *  rate-limit guardrails. */
  concurrency?: number
  onProgress?:  ProgressCallback
  signal?:      AbortSignal
}

export async function uploadChapterZip(
  client:     UploadHttpClient,
  materialId: number,
  zip:        Blob,
  opts:       UploadOptions = {},
): Promise<ApiChapterLike> {
  const concurrency = Math.max(1, opts.concurrency ?? DEFAULT_CONCURRENCY)
  const onProgress  = opts.onProgress ?? (() => {})

  // 1. Init.
  const init = await client.uploadInit(materialId, { byte_size: zip.size })
  if (init.parts.length === 0) {
    throw new Error('upload-init returned zero parts')
  }

  const tracker = new ProgressTracker(zip.size, init.parts.length, onProgress)
  tracker.setPhase('uploading')

  // 2. Drive parallel part PUTs. On any error, abort the multipart
  //    upload before re-throwing so the inbox key doesn't linger.
  const queue = [...init.parts]
  const completed: PutPartResult[] = []
  let failed: unknown = null

  const worker = async (): Promise<void> => {
    while (queue.length > 0 && failed === null) {
      const part = queue.shift()
      if (!part) return
      const start = (part.number - 1) * init.part_size
      const end   = Math.min(start + init.part_size, zip.size)
      const slice = zip.slice(start, end)
      try {
        const result = await putPart(part.number, part.url, slice, tracker, {
          signal: opts.signal,
        })
        completed.push(result)
      } catch (err) {
        failed = err
        return
      }
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()))

  if (failed !== null) {
    await safeAbort(client, materialId, init.tmp_id, init.upload_id)
    if (failed instanceof PermanentPutError) {
      throw new Error(`PUT ${failed.status}: ${failed.message}`)
    }
    throw failed
  }

  if (completed.length !== init.parts.length) {
    await safeAbort(client, materialId, init.tmp_id, init.upload_id)
    throw new Error(
      `upload incomplete: ${completed.length}/${init.parts.length} parts`,
    )
  }

  // 3. Finalize.
  tracker.setPhase('finalizing')
  tracker.flush()
  completed.sort((a, b) => a.number - b.number)
  const parts: FinalizePart[] = completed.map(c => ({
    number: c.number, etag: c.etag,
  }))

  try {
    return await client.uploadFinalize(materialId, {
      tmp_id:    init.tmp_id,
      upload_id: init.upload_id,
      parts,
      number:    opts.number,
      label:     opts.label,
    })
  } catch (err) {
    // Finalize failed (engine couldn't unpack/ingest, or quota tripped
    // at the commit point). Abort so the inbox key dies now.
    await safeAbort(client, materialId, init.tmp_id, init.upload_id)
    throw err
  }
}


async function safeAbort(
  client:     UploadHttpClient,
  materialId: number,
  tmp_id:     string,
  upload_id:  string,
): Promise<void> {
  try {
    await client.uploadAbort(materialId, { tmp_id, upload_id })
  } catch {
    // Best-effort. Bucket lifecycle rule sweeps anything we miss.
  }
}
