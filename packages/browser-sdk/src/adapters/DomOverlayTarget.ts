import type { TranslatedPage } from '../domain/translation'
import { attachOverlay, type OverlayOptions } from '../render/overlay'
import { imageAt, pageHostForImage, type DomPageOptions } from './domShared'

export type DomOverlayTargetOptions = DomPageOptions

export class DomOverlayTarget {
  private readonly imageSelector: string
  private readonly hostSelector?: string

  constructor(private readonly container: HTMLElement, options: DomOverlayTargetOptions = {}) {
    this.imageSelector = options.imageSelector ?? 'img'
    this.hostSelector = options.hostSelector
  }

  pageHost(index: number): HTMLElement {
    return pageHostForImage(imageAt(this.container, index, this.imageSelector), this.hostSelector, index)
  }

  attach(index: number, page: TranslatedPage, options?: OverlayOptions): HTMLElement {
    const host = this.pageHost(index)
    host.querySelector('[data-typoon-overlay="true"]')?.remove()
    return attachOverlay(host, page, options)
  }
}
