export interface BrowserModelHint {
  /** Model ID to use from the manifest. */
  readonly modelId: string
  /** Provider to prefer when creating the ORT session. */
  readonly preferredProvider: 'webgpu' | 'wasm'
  /** Number of wasm threads (1 for Safari / environments without SharedArrayBuffer). */
  readonly wasmNumThreads: number
}

export interface BrowserCapabilities {
  readonly supportsWebGpu: boolean
  readonly isSafari: boolean
  readonly modelHint: BrowserModelHint
}

/** Detect browser capabilities and return the optimal model hint. */
export function detectBrowserCapabilities(): BrowserCapabilities {
  const supportsWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator
  // Safari user-agent without Chrome (Chrome on iOS also reports "Safari" but includes "CriOS" or "Chrome")
  const isSafari = typeof navigator !== 'undefined'
    && /Safari/i.test(navigator.userAgent)
    && !/Chrome|CriOS/i.test(navigator.userAgent)

  const supportsCrossOriginIsolation = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated

  // Safari: wasm multi-threading crashes with SharedArrayBuffer.
  // Also force single-thread if crossOriginIsolated is not enabled.
  const wasmThreads = (isSafari || !supportsCrossOriginIsolation) ? 1
    : Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))

  return {
    supportsWebGpu,
    isSafari,
    modelHint: {
      modelId: supportsWebGpu ? 'comicDetr' : 'comicDetrWasm',
      preferredProvider: supportsWebGpu ? 'webgpu' : 'wasm',
      wasmNumThreads: wasmThreads,
    },
  }
}
