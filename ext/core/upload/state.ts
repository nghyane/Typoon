// Shared upload queue types between the SW (writer) and the popup
// (reader). The SW owns the queue under
// `chrome.storage.local[UPLOAD_QUEUE_KEY]`; the popup observes via
// storage.onChanged.
//
// Concurrency model: one running job at a time. Queueing many jobs
// is the user-visible feature; running them in parallel would race
// the engine quota gate and split inbox upload bandwidth pointlessly.

import type { ImageRef } from '@core/sources/extract'

export interface UploadJob {
  /** What the user picked. */
  images:    ImageRef[]
  projectId: number
  number?:   string
  title?:    string
  sourceUrl: string
  /** Display label (project title, set by the popup so the queue
   *  view doesn't have to look projects up itself). */
  projectTitle?: string
}

export type JobPhase =
  | 'queued'
  | 'fetching'    // pulling raw images from the source CDN
  | 'packing'    // building the zip from fetched bytes
  | 'uploading'  // multipart PUT to the inbox
  | 'finalizing' // engine is unpacking + ingesting
  | 'done'
  | 'error'

export interface QueuedJob {
  id:        string
  job:       UploadJob
  phase:     JobPhase
  /** Pages fetched from the source CDN so far. Total = job.images.length. */
  fetched:   number
  total:     number
  /** Bytes pushed to the inbox so far. Populated only while
   *  `phase === 'uploading'`. */
  bytesSent?:    number
  bytesTotal?:   number
  /** Bytes/second across the multipart PUT pool, EMA over ~3s.
   *  Undefined until the SDK has at least two progress samples. */
  speedBps?:     number
  /** Seconds remaining at the current speed. */
  etaSeconds?:   number
  /** Engine-assigned chapter number once finalize succeeds. */
  chapterNumber?: string
  /** Error message when phase === 'error'. */
  error?:    string
  enqueuedAt: number
  startedAt?: number
  finishedAt?: number
}

export interface UploadQueue {
  jobs: QueuedJob[]
}

export const UPLOAD_QUEUE_KEY = 'typoon.queue'

export const EMPTY_QUEUE: UploadQueue = { jobs: [] }

const ACTIVE_PHASES = new Set<JobPhase>(['queued', 'fetching', 'packing', 'uploading', 'finalizing'])
const RUNNING_PHASES = new Set<JobPhase>(['fetching', 'packing', 'uploading', 'finalizing'])

export function isJobActive(j: QueuedJob): boolean {
  return ACTIVE_PHASES.has(j.phase)
}

export function isJobRunning(j: QueuedJob): boolean {
  return RUNNING_PHASES.has(j.phase)
}

export function activeCount(q: UploadQueue): number {
  return q.jobs.filter(isJobActive).length
}

export function runningJob(q: UploadQueue): QueuedJob | undefined {
  return q.jobs.find(isJobRunning)
}
