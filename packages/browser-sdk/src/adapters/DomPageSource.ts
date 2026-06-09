import type { PageSource } from '../session'
import { imageAt, imagesIn, type DomPageOptions } from './domShared'

export type DomPageSourceOptions = DomPageOptions

export class DomPageSource implements PageSource {
  private readonly imageSelector: string

  constructor(private readonly container: HTMLElement, options: DomPageSourceOptions = {}) {
    this.imageSelector = options.imageSelector ?? 'img'
  }

  get pageCount(): number {
    return imagesIn(this.container, this.imageSelector).length
  }

  loadPage(index: number): HTMLImageElement {
    return imageAt(this.container, index, this.imageSelector)
  }
}
