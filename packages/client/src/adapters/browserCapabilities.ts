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
  readonly supportsStableWebGpu: boolean
  readonly isSafari: boolean
  readonly isIOS: boolean
  readonly isMobile: boolean
  readonly modelHint: BrowserModelHint
}

/** Detect browser capabilities and return the optimal model hint. */
export function detectBrowserCapabilities(): BrowserCapabilities {
  const supportsWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator
  // Safari user-agent without Chrome (Chrome on iOS also reports "Safari" but includes "CriOS" or "Chrome")
  const isSafari = typeof navigator !== 'undefined'
    && /Safari/i.test(navigator.userAgent)
    && !/Chrome|CriOS/i.test(navigator.userAgent)
  const isMobile = typeof navigator !== 'undefined'
    && (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || navigator.maxTouchPoints > 1)
  const isIOS = typeof navigator !== 'undefined'
    && (/iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1))
  const supportsStableWebGpu = supportsWebGpu && !isIOS

  const supportsCrossOriginIsolation = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated

  // Safari: wasm multi-threading crashes with SharedArrayBuffer.
  // Also force single-thread if crossOriginIsolated is not enabled.
  const wasmThreads = (isSafari || !supportsCrossOriginIsolation) ? 1
    : Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))

  return {
    supportsWebGpu,
    supportsStableWebGpu,
    isSafari,
    isIOS,
    isMobile,
    modelHint: {
      modelId: supportsStableWebGpu ? 'comicDetr' : 'comicDetrWasm',
      preferredProvider: supportsStableWebGpu ? 'webgpu' : 'wasm',
      wasmNumThreads: wasmThreads,
    },
  }
}
