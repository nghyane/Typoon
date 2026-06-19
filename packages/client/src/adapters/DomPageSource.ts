import type { PageSource } from '../domain/source'
import { imageAt, imagesIn, type DomPageOptions } from './domShared'

export type DomPageSourceOptions = DomPageOptions

export class DomPageSource implements PageSource {
  private readonly container: HTMLElement
  private readonly imageSelector: string

  constructor(container: HTMLElement, options: DomPageSourceOptions = {}) {
    this.container = container
    this.imageSelector = options.imageSelector ?? 'img'
  }

  get pageCount(): number {
    return imagesIn(this.container, this.imageSelector).length
  }

  loadPage(index: number): HTMLImageElement {
    return imageAt(this.container, index, this.imageSelector)
  }
}
