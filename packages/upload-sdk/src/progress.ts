// Progress aggregator for multipart uploads.
//
// Per-part XHR `upload.onprogress` fires byte deltas independently
// across N concurrent parts. The aggregator sums them into one
// monotonic counter and exposes a debounced snapshot the consumer can
// render at 4 fps (250 ms) without spamming React state updates or
// `chrome.storage.local` writes.
//
// Speed = EMA over a 3-second sample window. ETA = remaining/speed.

export interface UploadProgress {
  phase:       'packing' | 'uploading' | 'finalizing'
  bytesSent:   number
  bytesTotal:  number
  partsSent:   number
  partsTotal:  number
  /** Bytes/second over the last ~3 s. `undefined` for the first sample. */
  speedBps?:   number
  /** Seconds remaining at the current speed. `undefined` when speed is unknown. */
  etaSeconds?: number
}

export type ProgressCallback = (progress: UploadProgress) => void

const SPEED_WINDOW_MS = 3000
const EMIT_DEBOUNCE_MS = 250

export class ProgressTracker {
  private readonly bytesPerPart = new Map<number, number>()
  private readonly samples: { t: number; bytes: number }[] = []
  private partsSent = 0
  private phase: UploadProgress['phase'] = 'packing'
  private flushTimer: ReturnType<typeof setTimeout> | null = null
  private dirty = false

  private readonly bytesTotal: number
  private readonly partsTotal: number
  private readonly emit: ProgressCallback

  constructor(
    bytesTotal: number,
    partsTotal: number,
    emit: ProgressCallback,
  ) {
    this.bytesTotal = bytesTotal
    this.partsTotal = partsTotal
    this.emit = emit
    this.publishNow()
  }

  setPhase(phase: UploadProgress['phase']): void {
    this.phase = phase
    this.publishNow()
  }

  /** Called from XHR `upload.onprogress`. Delta is bytes since the
   *  last event for the same part number. Negative deltas (retry
   *  rollback) are accepted. */
  add(partNumber: number, deltaBytes: number): void {
    const cur = this.bytesPerPart.get(partNumber) ?? 0
    this.bytesPerPart.set(partNumber, Math.max(0, cur + deltaBytes))
    this.recordSample()
    this.queueEmit()
  }

  /** Reset a part's counter (used on retry). */
  reset(partNumber: number): void {
    this.bytesPerPart.set(partNumber, 0)
    this.queueEmit()
  }

  /** Mark a part as fully uploaded — clamps its byte count to the
   *  authoritative size (XHR sometimes drops the final delta) and
   *  bumps `partsSent`. */
  finalize(partNumber: number, partBytes: number): void {
    this.bytesPerPart.set(partNumber, partBytes)
    this.partsSent++
    this.recordSample()
    this.queueEmit()
  }

  flush(): void {
    if (this.flushTimer != null) {
      clearTimeout(this.flushTimer)
      this.flushTimer = null
    }
    this.publishNow()
  }

  private totalSent(): number {
    let total = 0
    for (const n of this.bytesPerPart.values()) total += n
    return Math.min(total, this.bytesTotal)
  }

  private recordSample(): void {
    const now = performance.now()
    this.samples.push({ t: now, bytes: this.totalSent() })
    while (this.samples.length > 1 && now - this.samples[0]!.t > SPEED_WINDOW_MS) {
      this.samples.shift()
    }
  }

  private speedBps(): number | undefined {
    if (this.samples.length < 2) return undefined
    const a = this.samples[0]!
    const b = this.samples[this.samples.length - 1]!
    const dt = (b.t - a.t) / 1000
    if (dt <= 0) return undefined
    const speed = (b.bytes - a.bytes) / dt
    return speed > 0 ? speed : undefined
  }

  private queueEmit(): void {
    this.dirty = true
    if (this.flushTimer != null) return
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null
      if (this.dirty) this.publishNow()
    }, EMIT_DEBOUNCE_MS)
  }

  private publishNow(): void {
    this.dirty = false
    const sent = this.totalSent()
    const speedBps = this.speedBps()
    const remaining = this.bytesTotal - sent
    const etaSeconds = speedBps && speedBps > 0
      ? Math.max(0, Math.round(remaining / speedBps))
      : undefined
    this.emit({
      phase: this.phase,
      bytesSent: sent,
      bytesTotal: this.bytesTotal,
      partsSent: this.partsSent,
      partsTotal: this.partsTotal,
      speedBps,
      etaSeconds,
    })
  }
}
