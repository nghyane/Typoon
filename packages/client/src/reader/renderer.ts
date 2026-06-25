// reader/renderer.ts — the port the translation controller renders through.
//
// The controller produces overlay DATA (per-page ReaderPageOverlay) and hands it
// to a renderer; it never builds overlay DOM itself. This keeps the controller
// framework-agnostic: the web app can inject a Svelte-native renderer and a
// future extension its own, both implementing this interface.

import type { ReaderPageOverlay } from '../domain/pageScan'

export interface OverlayChapterMeta {
  readonly sourceLanguage: string | null
  readonly targetLanguage: string | null
}

export interface ReaderRenderer {
  /** The chapter content host the overlays/seams attach to (read for layout). */
  readonly currentHost: HTMLElement | null
  setHost(host: HTMLElement | null): void
  /** Register a page element; returns cleanup. */
  registerPage(pageIndex: number, el: HTMLElement): () => void
  /** Replace overlay data; the renderer reconciles what changed. */
  update(overlays: Map<number, ReaderPageOverlay>, revision: number, meta: OverlayChapterMeta): void
  setHidden(hidden: boolean): void
  detach(): void
  dispose(): void
}
