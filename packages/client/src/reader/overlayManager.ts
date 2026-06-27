// reader/overlayManager.ts — owns overlay DOM attach/detach and visibility.
//
// Two render surfaces:
//   - Page surface: per-page overlay attached to each [data-page-index] frame.
//     The frame is aspect-locked to the source page and declares
//     container-type:inline-size, so overlay geometry (% of frame) coincides
//     with the displayed image and typography scales via cqw — no JS measuring.
//     Holds page-local bubbles whose bbox stays inside the page.
//   - Seam bridge: a bubble that crosses the gap between two stacked pages is
//     rendered on a bridge element positioned on the chapter-level host (which
//     spans the whole strip), so it is not clipped by either page. The bridge is
//     positioned from the owning page's live rect; it spans into the neighbor.
//
// Performance: overlays self-scale, so once attached they stay correct without
// re-attachment. Visibility tracked via IntersectionObserver (no per-frame
// reflow); each page attaches once per revision. Seams reposition on every
// visibility change (cheap) and on host resize.

import { attachOverlay } from '../render/overlay'
import type { ReaderPageOverlay, SeamOverlay } from '../domain/pageScan'
import type { OverlayChapterMeta, ReaderRenderer } from './renderer'

export type { OverlayChapterMeta }

interface AttachedOverlay {
  readonly el: HTMLElement
}

interface SeamBridge {
  readonly el: HTMLElement
  readonly ownerPageIndex: number
  readonly side: 'below' | 'above'
  readonly seam: SeamOverlay
}

export class OverlayManager implements ReaderRenderer {
  private host: HTMLElement | null = null
  private overlays = new Map<number, ReaderPageOverlay>()
  /** Overlay data that each page was last attached with (object identity check). */
  private readonly attachedData = new Map<number, ReaderPageOverlay>()
  private meta: OverlayChapterMeta = { sourceLanguage: null, targetLanguage: null }
  private hidden = false

  private readonly pageEls = new Map<number, HTMLElement>()
  private readonly attached = new Map<number, AttachedOverlay>()
  private readonly seams = new Map<string, SeamBridge>()
  private readonly visible = new Set<number>()
  private observer: IntersectionObserver | null = null
  private hostObserver: ResizeObserver | null = null

  constructor(private readonly marginPx: number) {}

  setHost(host: HTMLElement | null): void {
    this.hostObserver?.disconnect()
    this.hostObserver = null
    this.host = host
    if (!host) return
    if ('ResizeObserver' in window) {
      this.hostObserver = new ResizeObserver(() => this.repositionSeams())
      this.hostObserver.observe(host)
    }
    this.attachVisible()
  }

  get currentHost(): HTMLElement | null {
    return this.host
  }

  /** Register a page element (called from a Svelte action). Returns cleanup. */
  registerPage(pageIndex: number, el: HTMLElement): () => void {
    this.pageEls.set(pageIndex, el)
    this.ensureObserver()
    this.observer?.observe(el)
    if (!this.observer) this.visible.add(pageIndex)
    this.attachVisible()
    return () => {
      this.observer?.unobserve(el)
      this.pageEls.delete(pageIndex)
      this.visible.delete(pageIndex)
      const stored = this.attached.get(pageIndex)
      if (stored) {
        stored.el.remove()
        this.attached.delete(pageIndex)
        this.attachedData.delete(pageIndex)
      }
      this.removeSeamsOwnedBy(pageIndex)
    }
  }

  /** Update overlay data; re-attaches only pages whose data changed. */
  update(overlays: Map<number, ReaderPageOverlay>, _revision: number, meta: OverlayChapterMeta): void {
    this.overlays = overlays
    this.meta = meta
    this.attachVisible()
  }

  setHidden(hidden: boolean): void {
    this.hidden = hidden
    for (const { el } of this.attached.values()) el.style.display = hidden ? 'none' : ''
    for (const { el } of this.seams.values()) el.style.display = hidden ? 'none' : ''
  }

  detach(): void {
    for (const { el } of this.attached.values()) el.remove()
    this.attached.clear()
    this.attachedData.clear()
    for (const { el } of this.seams.values()) el.remove()
    this.seams.clear()
    // Drop the data source too. Otherwise a later attachVisible() — fired by the
    // IntersectionObserver while scrolling the next chapter, or by page
    // re-registration — would resurrect these overlays onto the new pages, since
    // detach() only removed the DOM, not the map it re-renders from.
    this.overlays = new Map()
  }

  dispose(): void {
    this.detach()
    this.observer?.disconnect()
    this.observer = null
    this.hostObserver?.disconnect()
    this.hostObserver = null
    this.pageEls.clear()
    this.attachedData.clear()
    this.visible.clear()
    this.host = null
  }

  private ensureObserver(): void {
    if (this.observer || typeof IntersectionObserver === 'undefined') return
    this.observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          const index = Number((entry.target as HTMLElement).dataset.pageIndex)
          if (!Number.isFinite(index)) continue
          if (entry.isIntersecting) this.visible.add(index)
          else this.visible.delete(index)
        }
        this.attachVisible()
      },
      { rootMargin: `${this.marginPx}px 0px` },
    )
    for (const el of this.pageEls.values()) this.observer.observe(el)
  }

  private attachVisible(): void {
    if (!this.host?.isConnected) return
    const targets = this.observer ? this.visible : new Set(this.pageEls.keys())
    for (const pageIndex of targets) {
      const overlay = this.overlays.get(pageIndex)
      if (!overlay) continue
      this.attachPageSurface(pageIndex, overlay)
      this.attachSeam(`${pageIndex}:below`, overlay.seamBelow, pageIndex, 'below')
      this.attachSeam(`${pageIndex}:above`, overlay.seamAbove, pageIndex, 'above')
    }
    this.repositionSeams()
  }

  private attachPageSurface(pageIndex: number, overlay: ReaderPageOverlay): void {
    const stored = this.attached.get(pageIndex)
    // Object identity: same overlay object = nothing changed for this page.
    if (stored && overlay === this.attachedData.get(pageIndex)) return
    const pageEl = this.pageEls.get(pageIndex)
    if (!pageEl) return
    if (!overlay.items.length) {
      if (stored) {
        stored.el.remove()
        this.attached.delete(pageIndex)
        this.attachedData.delete(pageIndex)
      }
      return
    }
    if (stored) stored.el.remove()
    const overlayEl = attachOverlay(pageEl, {
      pageSize: overlay.pageSize,
      placements: overlay.items.map(item => item.placement),
      translations: overlay.translations,
      placementMargins: overlay.items.map(item => item.margin),
      fontContextPlacements: overlay.items.map(item => item.placement),
      sourceLanguage: this.meta.sourceLanguage,
      targetLanguage: this.meta.targetLanguage,
    })
    if (this.hidden) overlayEl.style.display = 'none'
    this.attached.set(pageIndex, { el: overlayEl })
    this.attachedData.set(pageIndex, overlay)
  }

  private attachSeam(key: string, seam: SeamOverlay | null, ownerPageIndex: number, side: 'below' | 'above'): void {
    const existing = this.seams.get(key)
    if (!seam || !seam.items.length) {
      if (existing) {
        existing.el.remove()
        this.seams.delete(key)
      }
      return
    }
    if (existing && existing.seam === seam) return
    if (existing) existing.el.remove()

    const wrapper = document.createElement('div')
    wrapper.dataset.typoonSeam = key
    wrapper.style.position = 'absolute'
    wrapper.style.pointerEvents = 'none'
    wrapper.style.zIndex = '1'
    // The seam bridge is its own positioned element (not inside a page frame),
    // so it must establish the container that the overlay's cqw typography
    // resolves against. Its width is kept equal to the displayed page width by
    // positionBridge, so 100cqw === displayed page width === seam pageW * scale.
    wrapper.style.containerType = 'inline-size'
    const overlayEl = attachOverlay(wrapper, {
      pageSize: seam.seamSize,
      placements: seam.items.map(item => item.placement),
      translations: seam.translations,
      placementMargins: seam.items.map(item => item.margin),
      fontContextPlacements: seam.items.map(item => item.placement),
      sourceLanguage: this.meta.sourceLanguage,
      targetLanguage: this.meta.targetLanguage,
    })
    overlayEl.style.position = 'absolute'
    overlayEl.style.inset = '0'
    if (this.hidden) wrapper.style.display = 'none'
    this.host?.appendChild(wrapper)
    const bridge: SeamBridge = { el: wrapper, ownerPageIndex, side, seam }
    this.seams.set(key, bridge)
    this.positionBridge(bridge)
  }

  private repositionSeams(): void {
    for (const bridge of this.seams.values()) this.positionBridge(bridge)
  }

  /**
   * Position a seam bridge from its owning page's live rect.
   * seamSize is in the owner page's source px; the owner occupies
   *   [topOffsetSource, topOffsetSource + ownerSourceHeight) of the seam band.
   * displayScale is derived from the owner's display width (all pages share it).
   */
  private positionBridge(bridge: SeamBridge): void {
    const host = this.host
    const ownerEl = this.pageEls.get(bridge.ownerPageIndex)
    if (!host?.isConnected || !ownerEl) return
    const [seamW, seamH] = bridge.seam.seamSize
    const topOffsetSource = bridge.side === 'below' ? 0 : bridge.seam.seamSplitY
    const hostRect = host.getBoundingClientRect()
    const ownerRect = ownerEl.getBoundingClientRect()
    const displayScale = ownerRect.width / Math.max(1, seamW)
    bridge.el.style.left = `${ownerRect.left - hostRect.left}px`
    bridge.el.style.top = `${ownerRect.top - hostRect.top - topOffsetSource * displayScale}px`
    bridge.el.style.width = `${ownerRect.width}px`
    bridge.el.style.height = `${seamH * displayScale}px`
  }

  private removeSeamsOwnedBy(pageIndex: number): void {
    for (const [key, bridge] of this.seams) {
      if (bridge.ownerPageIndex !== pageIndex) continue
      bridge.el.remove()
      this.seams.delete(key)
    }
  }
}
