// Multipart-part PUT with progress + retry.
//
// Two transport paths, picked at runtime:
//   - XMLHttpRequest when available (popup, content script, SPA).
//     Gives byte-level upload progress via `xhr.upload.onprogress`,
//     which `fetch` still lacks in stable Chrome.
//   - `fetch` fallback for MV3 service workers — they have no XHR
//     constructor at all. We lose mid-part progress and only emit
//     once the part lands; the per-part counter still ticks so the
//     UI shows movement at part boundaries.
//
// Each PUT is a single presigned URL; the server (R2/S3/local)
// returns the part's ETag in the `ETag` response header, which we
// hand back to `upload-finalize`.
//
// Retry policy: 3 attempts, exponential backoff (250ms → 1s → 4s),
// only on transient failures (network error or 5xx). 4xx is
// permanent (auth/signature/CORS misconfig — retrying won't help).
// On retry, the progress tracker is reset for that part so the
// aggregate counter doesn't double-count.

import type { ProgressTracker } from './progress'

export interface PutPartResult {
  number: number
  etag:   string
  bytes:  number
}

export interface PutPartOptions {
  /** Cooperative cancellation. Aborting an in-flight PUT cancels the XHR. */
  signal?:   AbortSignal
  /** Per-part retry budget. Default 3. */
  attempts?: number
}

const DEFAULT_ATTEMPTS = 3

export async function putPart(
  partNumber: number,
  url:        string,
  body:       Blob,
  tracker:    ProgressTracker,
  opts:       PutPartOptions = {},
): Promise<PutPartResult> {
  const attempts = Math.max(1, opts.attempts ?? DEFAULT_ATTEMPTS)
  let lastErr: unknown

  for (let attempt = 1; attempt <= attempts; attempt++) {
    if (opts.signal?.aborted) throw new DOMException('aborted', 'AbortError')

    if (attempt > 1) {
      // Roll back this part's progress before retrying so the global
      // counter stays accurate.
      tracker.reset(partNumber)
      const backoffMs = 250 * Math.pow(4, attempt - 2)   // 250, 1000, 4000
      await sleep(backoffMs, opts.signal)
    }

    try {
      const etag = await putOnce(partNumber, url, body, tracker, opts.signal)
      tracker.finalize(partNumber, body.size)
      return { number: partNumber, etag, bytes: body.size }
    } catch (err) {
      lastErr = err
      if (err instanceof PermanentPutError) throw err
      if ((err as { name?: string })?.name === 'AbortError') throw err
      // else: transient — retry
    }
  }

  throw lastErr ?? new Error(`PUT part ${partNumber} failed`)
}


// ── Internals ─────────────────────────────────────────────────────


export class PermanentPutError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'PermanentPutError'
  }
}

function putOnce(
  partNumber: number,
  url:        string,
  body:       Blob,
  tracker:    ProgressTracker,
  signal:     AbortSignal | undefined,
): Promise<string> {
  // MV3 service workers have no `XMLHttpRequest`; fall back to fetch.
  // Popup / content / SPA contexts get the XHR path which can stream
  // upload progress.
  if (typeof XMLHttpRequest === 'undefined') {
    return putOnceFetch(partNumber, url, body, tracker, signal)
  }
  return putOnceXhr(partNumber, url, body, tracker, signal)
}


function putOnceXhr(
  partNumber: number,
  url:        string,
  body:       Blob,
  tracker:    ProgressTracker,
  signal:     AbortSignal | undefined,
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url, true)

    let lastLoaded = 0
    xhr.upload.onprogress = (e) => {
      // `e.loaded` is monotonic over a single request; convert to delta
      // so the tracker can sum across concurrent parts.
      const delta = e.loaded - lastLoaded
      lastLoaded = e.loaded
      if (delta > 0) tracker.add(partNumber, delta)
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
        // R2/S3 quote the ETag; the API expects the raw hex.
        resolve(raw.replace(/^"|"$/g, ''))
      } else if (xhr.status >= 400 && xhr.status < 500) {
        reject(new PermanentPutError(xhr.status, `PUT ${xhr.status}`))
      } else {
        reject(new Error(`PUT ${xhr.status}`))
      }
    }
    xhr.onerror = () => reject(new Error('network'))
    xhr.ontimeout = () => reject(new Error('timeout'))

    if (signal) {
      if (signal.aborted) {
        xhr.abort()
        reject(new DOMException('aborted', 'AbortError'))
        return
      }
      signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }

    xhr.send(body)
  })
}


async function putOnceFetch(
  partNumber: number,
  url:        string,
  body:       Blob,
  tracker:    ProgressTracker,
  signal:     AbortSignal | undefined,
): Promise<string> {
  // No mid-part progress in this branch — `Request` upload streaming
  // is gated behind `duplex: 'half'` and isn't reliable on the SW
  // path. We emit once the body is accepted; `finalize()` in the
  // outer retry wrapper still settles the per-part total.
  const res = await fetch(url, {
    method: 'PUT',
    body,
    signal,
  })

  if (res.status >= 200 && res.status < 300) {
    tracker.add(partNumber, body.size)
    const raw = res.headers.get('ETag')
    if (!raw) {
      throw new PermanentPutError(
        res.status,
        'Missing ETag — kiểm tra CORS bucket (ExposeHeaders: ETag).',
      )
    }
    return raw.replace(/^"|"$/g, '')
  }
  if (res.status >= 400 && res.status < 500) {
    throw new PermanentPutError(res.status, `PUT ${res.status}`)
  }
  throw new Error(`PUT ${res.status}`)
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
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
