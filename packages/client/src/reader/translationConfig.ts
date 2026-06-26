// reader/translationConfig.ts — injectable configuration for ReaderTranslation.
// Replaces hardcoded module-level constants so the engine is testable/configurable.

import type { OrtProvider } from '../models/OrtSessionPool'

export interface TranslationConfig {
  readonly model: { readonly repo: string; readonly revision: string; readonly proxyBase: string }
  readonly chunk: { readonly overlayMarginPx: number; readonly processMarginPx: number }
  readonly scan: ScanConfig
  readonly resilience: { readonly maxChunkAttempts: number; readonly backoffMs: number }
  readonly memory: { readonly maxCachedPages: number }
  /** Run Comic-DETR inference in a Web Worker to keep the main thread free. */
  readonly detector: { readonly useWorker: boolean }
  readonly translator: {
    readonly maxSessionsDesktop: number
    readonly maxSessionsMobile: number
    /** Pages processed concurrently (OCR of one overlaps DeepL of another). */
    readonly maxPagesInFlightDesktop: number
    readonly maxPagesInFlightMobile: number
  }
}

export interface ScanConfig {
  /** Cap OCR canvas width (source px) to bound recognition/detection cost. */
  readonly maxCaptureWidth: number
  /** Halo height as a fraction of the page's source height. */
  readonly haloRatio: number
  /** Absolute cap on halo height (source px). */
  readonly haloMaxPx: number
}

const DISCORD_CDN_PROXY_BASE = 'https://927251094806098001.discordsays.com/cdn/c'

export const defaultTranslationConfig: TranslationConfig = {
  model: { repo: 'nghyane/comic-detr', revision: 'v1', proxyBase: DISCORD_CDN_PROXY_BASE },
  chunk: { overlayMarginPx: 1400, processMarginPx: 450 },
  scan: { maxCaptureWidth: 1280, haloRatio: 0.25, haloMaxPx: 600 },
  resilience: { maxChunkAttempts: 3, backoffMs: 400 },
  memory: { maxCachedPages: 12 },
  detector: { useWorker: false },
  translator: { maxSessionsDesktop: 3, maxSessionsMobile: 1, maxPagesInFlightDesktop: 3, maxPagesInFlightMobile: 1 },
}

export function preferredProviders(preferred: OrtProvider): readonly OrtProvider[] {
  return preferred === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
}
