import type { PreparedChapter } from '../domain/preparedChapter'
import type { PageDocumentSource } from '../domain/source'

export class PreparedChapterSource implements PageDocumentSource {
  readonly pageCount: number
  private readonly chapter: PreparedChapter

  constructor(chapter: PreparedChapter) {
    this.chapter = chapter
    this.pageCount = chapter.pages.length
  }

  async readPage(index: number, signal?: AbortSignal) {
    const page = this.chapter.pages[index]
    if (!page) throw new RangeError(`prepared page ${index} out of range`)
    return {
      index,
      pixels: await page.asset.readPixels(signal),
      size: page.size,
      projections: page.projections,
    }
  }
}
