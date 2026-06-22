// reader/asyncSignal.ts — abort + idle-scheduling helpers shared by the reader
// stage layer. One concern only (signal handling and main-thread yielding); not
// a generic util bucket.

/** Throw signal.reason (or a generic Error) if the signal is already aborted. */
export function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

/** Normalize any thrown value to a message string. */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

let neverAbortSignal: AbortSignal | null = null

/** A signal that never aborts — for background work that must outlive a run. */
export function neverAbort(): AbortSignal {
  if (!neverAbortSignal) neverAbortSignal = new AbortController().signal
  return neverAbortSignal
}

interface IdleGlobal {
  requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number
}

/** Run once the main thread is idle (requestIdleCallback, setTimeout fallback). */
export function whenIdle(run: () => void, timeoutMs = 2000): void {
  const win = globalThis as typeof globalThis & IdleGlobal
  if (typeof win.requestIdleCallback === 'function') win.requestIdleCallback(run, { timeout: timeoutMs })
  else setTimeout(run, 0)
}

/** Promise that resolves when the main thread is idle. */
export function yieldToIdle(timeoutMs: number): Promise<void> {
  return new Promise<void>(resolve => whenIdle(() => resolve(), timeoutMs))
}
