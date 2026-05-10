// Background service worker — owns the upload queue.
//
// One queue, one running job at a time. New jobs go to the back of the
// queue; the worker loop picks them off as the running slot frees up.
// Each job's phase + progress lives in chrome.storage.local under
// `typoon.queue` so the popup can render the list without a long-lived
// message channel.
//
// Pipeline per job:
//   1. fetching   — pull every CDN image in parallel (HTTP/2 multiplex)
//   2. packing    — fflate store-mode zip of all fetched bytes
//   3. uploading  — `@typoon/upload-sdk` multipart PUT to the inbox
//   4. finalizing — engine downloads zip, prepares chapter, ingests
//
// Why one-at-a-time rather than N parallel?
//   - Engine quota is shared across all in-flight uploads; running two
//     in parallel doubles the chance of one tripping the gate
//     mid-flight.
//   - Inbox bandwidth split N ways = same total throughput, longer
//     per-job latency. Sequential uploads finish individually fast;
//     the user gets the first "done" notification sooner.
//   - SW lifecycle: any one job that hits the 5-min hard cap takes the
//     whole worker down. Sequential = blast radius of one job.
//
// Listener registration discipline: register synchronously at the top
// of `defineBackground` and never throw. Anything async runs inside
// handlers wrapped in try/catch so a busted handler doesn't brick the
// message channel.

import { TypoonClient, QuotaExceededError } from '@core/typoon'
import { API_URL } from '@core/config'
import { chromeStorage } from '@shell/adapters/chrome-storage'
import {
  packPagesToZip, uploadChapterZip, type UploadProgress,
} from '@typoon/upload-sdk'
import {
  EMPTY_QUEUE, UPLOAD_QUEUE_KEY,
  isJobRunning, type QueuedJob, type UploadJob, type UploadQueue,
} from '@core/upload/state'

export default defineBackground(() => {
  console.log('[typoon-bg v6] booted (zip + multipart inbox)', { id: browser.runtime.id })

  browser.runtime.onMessage.addListener((raw, sender, respond) => {
    try {
      return handleMessage(raw, sender, respond)
    } catch (err) {
      console.error('[typoon-bg] listener threw', err)
      try { respond({ ok: false, error: String(err) }) } catch {}
      return false
    }
  })

  // Reset on boot. The SW just woke up from cold; any job in a running
  // phase was interrupted (browser closed mid-run, OOM, manual disable).
  // Mark them errored so the user can retry.
  void resetStaleJobs().then(() => {
    void kickWorker()
  })
})

function handleMessage(
  raw: unknown,
  sender: Browser.runtime.MessageSender,
  respond: (v: unknown) => void,
): true | false {
  const msg = raw as { type?: string; job?: UploadJob; jobId?: string }

  if (msg?.type === 'tab/whoami') {
    respond({ ok: true, tabId: sender.tab?.id ?? 0 })
    return false
  }

  if (msg?.type === 'queue/enqueue' && msg.job) {
    const job = msg.job
    enqueue(job)
      .then(id => {
        respond({ ok: true, id })
        void kickWorker()
      })
      .catch(err => {
        try { respond({ ok: false, error: String(err) }) } catch {}
      })
    return true
  }

  if (msg?.type === 'queue/dismiss' && msg.jobId) {
    void dismissJob(msg.jobId).then(() => respond({ ok: true }))
    return true
  }

  if (msg?.type === 'queue/cancel' && msg.jobId) {
    // Only cancels QUEUED jobs — cancelling a running upload mid-PUT
    // is messy and probably not what the user wants. The bucket
    // lifecycle rule sweeps anything left half-uploaded.
    void cancelJob(msg.jobId).then(ok => respond({ ok }))
    return true
  }

  if (msg?.type === 'queue/retry' && msg.jobId) {
    void retryJob(msg.jobId).then(ok => {
      respond({ ok })
      if (ok) void kickWorker()
    })
    return true
  }

  // Bulk delete handlers used to live here; the popup no longer
  // exposes "Xoá lịch sử" / "Xoá tất cả" because the SW now
  // auto-evicts done jobs after 24h. If a future surface needs
  // these, re-add them — keep the type guard symmetrical with the
  // popup affordance, never the other way round.

  respond({ ok: false, error: `unknown message: ${msg?.type}` })
  return false
}

// ── Queue mutations ─────────────────────────────────────────────────

// Done jobs are kept for a short window so the user can see what
// they just upload-ed when they reopen the popup; after that they're
// noise. Auto-purge on every read keeps the queue lean without
// requiring a UI action ("Xoá lịch sử" used to live in the popup
// but bulk-delete is a workflow no real user asks for).
// Auto-prune rules applied on every readQueue:
//
//   1. TTL — done > 24h or error > 7 days are dropped. Done is the
//      shorter window because users only need confirmation that the
//      last upload landed; error stays longer so users coming back
//      after a weekend still see what failed.
//   2. Cap — at most MAX_ROWS visible total. When over, drop the
//      oldest dismissable rows (done first, then error) by
//      finishedAt; running/queued never get evicted because they
//      represent live work.
//
// Both rules run on read so the popup is never the one paying. The
// list grows during a busy session and self-heals between sessions
// without any UI action.

const DONE_RETENTION_MS  =       24 * 60 * 60 * 1000   // 24h
const ERROR_RETENTION_MS =   7 * 24 * 60 * 60 * 1000   // 7 days
const MAX_ROWS = 20

async function readQueue(): Promise<UploadQueue> {
  const q   = (await chromeStorage.get<UploadQueue>(UPLOAD_QUEUE_KEY)) ?? EMPTY_QUEUE
  const now = Date.now()

  // Stage 1 — TTL purge.
  const live = q.jobs.filter(j => {
    if (j.phase === 'done') {
      return now - (j.finishedAt ?? j.enqueuedAt) < DONE_RETENTION_MS
    }
    if (j.phase === 'error') {
      return now - (j.finishedAt ?? j.enqueuedAt) < ERROR_RETENTION_MS
    }
    return true   // running / queued / fetching / packing / uploading / finalizing
  })

  // Stage 2 — cap. Only dismissable rows can be evicted; running
  // work always survives. Oldest goes first within done/error.
  let kept = live
  if (live.length > MAX_ROWS) {
    const dismissable = live
      .map((j, idx) => ({ j, idx }))
      .filter(({ j }) => j.phase === 'done' || j.phase === 'error')
      .sort((a, b) => {
        // Done evicted before error (shorter relevance), then by age.
        if (a.j.phase !== b.j.phase) return a.j.phase === 'done' ? -1 : 1
        const at = a.j.finishedAt ?? a.j.enqueuedAt
        const bt = b.j.finishedAt ?? b.j.enqueuedAt
        return at - bt
      })
    const drop = new Set<number>()
    let need = live.length - MAX_ROWS
    for (const { idx } of dismissable) {
      if (need <= 0) break
      drop.add(idx)
      need--
    }
    kept = live.filter((_, idx) => !drop.has(idx))
  }

  if (kept.length !== q.jobs.length) {
    const trimmed: UploadQueue = { jobs: kept }
    await chromeStorage.set(UPLOAD_QUEUE_KEY, trimmed)
    return trimmed
  }
  return q
}

async function writeQueue(q: UploadQueue): Promise<void> {
  await chromeStorage.set(UPLOAD_QUEUE_KEY, q)
}

async function enqueue(job: UploadJob): Promise<string> {
  const id = `j-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  const q = await readQueue()
  q.jobs.push({
    id,
    job,
    phase:      'queued',
    fetched:    0,
    total:      job.images.length,
    enqueuedAt: Date.now(),
  })
  await writeQueue(q)
  return id
}

async function updateJob(id: string, patch: Partial<QueuedJob>): Promise<void> {
  const q = await readQueue()
  const idx = q.jobs.findIndex(j => j.id === id)
  if (idx < 0) return
  q.jobs[idx] = { ...q.jobs[idx]!, ...patch }
  await writeQueue(q)
}

async function dismissJob(id: string): Promise<void> {
  const q = await readQueue()
  q.jobs = q.jobs.filter(j => j.id !== id)
  await writeQueue(q)
}

async function cancelJob(id: string): Promise<boolean> {
  const q = await readQueue()
  const idx = q.jobs.findIndex(j => j.id === id)
  if (idx < 0) return false
  if (q.jobs[idx]!.phase !== 'queued') return false
  q.jobs.splice(idx, 1)
  await writeQueue(q)
  return true
}

async function retryJob(id: string): Promise<boolean> {
  const q = await readQueue()
  const idx = q.jobs.findIndex(j => j.id === id)
  if (idx < 0) return false
  if (q.jobs[idx]!.phase !== 'error') return false
  q.jobs[idx] = {
    ...q.jobs[idx]!,
    phase: 'queued',
    error: undefined,
    chapterNumber: undefined,
    finishedAt: undefined,
    fetched: 0,
  }
  await writeQueue(q)
  return true
}

async function resetStaleJobs(): Promise<void> {
  const q = await readQueue()
  let dirty = false
  for (const j of q.jobs) {
    if (isJobRunning(j)) {
      j.phase = 'error'
      j.error = 'Bị gián đoạn. Hãy thử lại.'
      j.finishedAt = Date.now()
      dirty = true
    }
  }
  if (dirty) await writeQueue(q)
}

// ── Worker loop ─────────────────────────────────────────────────────

let workerActive = false

async function kickWorker(): Promise<void> {
  if (workerActive) return
  workerActive = true
  try {
    while (true) {
      const q = await readQueue()
      const next = q.jobs.find(j => j.phase === 'queued')
      if (!next) break
      await runJob(next.id)
    }
  } finally {
    workerActive = false
  }
}

async function runJob(id: string): Promise<void> {
  const q0 = await readQueue()
  const job0 = q0.jobs.find(j => j.id === id)
  if (!job0) return
  const job = job0.job
  const startedAt = Date.now()

  await updateJob(id, { phase: 'fetching', fetched: 0, startedAt })

  try {
    const token = await readToken()
    if (!token) throw new Error('Chưa đăng nhập.')
    const client = new TypoonClient({ apiUrl: API_URL, token })

    // 1. Fetch all CDN bytes in parallel. HTTP/2 multiplexes these
    //    onto one connection; manga CDNs are usually the bottleneck so
    //    we let the browser scheduler handle ordering.
    const sourceOrigin = (() => {
      try { return new URL(job.sourceUrl).origin + '/' } catch { return undefined }
    })()
    let fetched = 0
    const fetchedBlobs = await Promise.all(job.images.map(async (img) => {
      const blob = await fetchImage(img.url, sourceOrigin)
      fetched++
      // Persist every 4 pages to keep storage writes off the hot path.
      if (fetched === job.images.length || fetched % 4 === 0) {
        await updateJob(id, { fetched })
      }
      return { source: img.url, bytes: new Uint8Array(await blob.arrayBuffer()) }
    }))
    await updateJob(id, { fetched: job.images.length, phase: 'packing' })

    // 2. Pack into store-mode zip. Synchronous; ~50ms for 30 pages.
    const zip = packPagesToZip(fetchedBlobs)

    // 3. Multipart PUT via the shared SDK. Per-byte progress is no
    //    longer surfaced (popup shows three buckets: running / done
    //    / error), so we only flip the phase when finalize starts.
    //    One storage write per phase boundary instead of N at 4fps.
    let lastPhase: UploadProgress['phase'] | null = null
    await updateJob(id, { phase: 'uploading' })
    const chapter = await uploadChapterZip(client, job.projectId, zip, {
      number: job.number,
      title:  job.title,
      onProgress: (p: UploadProgress) => {
        if (p.phase === lastPhase) return
        lastPhase = p.phase
        void updateJob(id, { phase: p.phase })
      },
    })

    await updateJob(id, {
      phase: 'done',
      chapterNumber: chapter.number,
      finishedAt: Date.now(),
    })

    notifyDone(job, chapter.number)
  } catch (err) {
    let message = (err as Error)?.message ?? String(err)
    if (err instanceof QuotaExceededError) {
      message = 'Đã đạt giới hạn upload. Hãy thử lại sau.'
    }
    console.error('[typoon-bg] job failed', id, err)
    await updateJob(id, {
      phase: 'error',
      error: message,
      finishedAt: Date.now(),
    })
  }
}

function notifyDone(job: UploadJob, chapterNumber: string) {
  try {
    const iconUrl = (browser.runtime.getURL as (p: string) => string)('/icon/128.png')
    const project = job.projectTitle ? ` • ${job.projectTitle}` : ''
    void browser.notifications.create({
      type: 'basic',
      iconUrl,
      title: 'Hội Mê Truyện — đã upload',
      message: `Chương ${chapterNumber}${project} (${job.images.length} ảnh)`,
    })
  } catch {}
}

async function readToken(): Promise<string | null> {
  const cfg = await chromeStorage.get<{ token?: string }>('typoon.config')
  return cfg?.token ?? null
}

// ── CDN image fetch ────────────────────────────────────────────────

async function fetchImage(url: string, sourceOrigin: string | undefined): Promise<Blob> {
  // Attempt 1: page-origin referrer (covers hotlink-protected CDNs).
  try {
    const res = await fetch(url, {
      cache: 'no-store',
      referrer: sourceOrigin,
      referrerPolicy: 'strict-origin-when-cross-origin',
    })
    if (res.ok) return await res.blob()
    if (res.status !== 403) throw new Error(`HTTP ${res.status}`)
  } catch {
    // fall through
  }
  // Attempt 2: no referrer (covers strict CDNs).
  const res2 = await fetch(url, { cache: 'no-store', referrerPolicy: 'no-referrer' })
  if (!res2.ok) throw new Error(`HTTP ${res2.status} (${url.slice(0, 60)}…)`)
  return await res2.blob()
}
