import type { RenderedPage } from '../domain/translation'
import { EventBuffer } from './EventBuffer'
import type { SegmentEvent, SegmentEventListener } from './events'
import { PageDataflow, type PageDataflowOptions } from './PageDataflow'

export interface SegmentTranslationRunOptions extends Omit<PageDataflowOptions, 'pageIndex' | 'signal' | 'emit'> {
  readonly pages?: readonly number[]
  readonly stopOnError?: boolean
  readonly signal?: AbortSignal
}

export class SegmentTranslationRun {
  private readonly eventBuffer = new EventBuffer<SegmentEvent>()
  private readonly abortController = new AbortController()
  private readonly latestPages = new Map<number, RenderedPage>()
  private readonly donePromise: Promise<readonly RenderedPage[]>

  constructor(private readonly options: SegmentTranslationRunOptions) {
    this.donePromise = this.run()
  }

  subscribe(listener: SegmentEventListener): () => void {
    return this.eventBuffer.subscribe(listener)
  }

  events(): AsyncIterable<SegmentEvent> {
    return this.eventBuffer.events()
  }

  done(): Promise<readonly RenderedPage[]> {
    return this.donePromise
  }

  cancel(reason = new Error('segment translation cancelled')): void {
    if (!this.abortController.signal.aborted) this.abortController.abort(reason)
    this.emit({ type: 'segment-cancelled' })
    this.eventBuffer.close()
  }

  private async run(): Promise<readonly RenderedPage[]> {
    const pages = this.options.pages ?? range(this.options.source.pageCount)
    const signal = this.combinedSignal()
    this.emit({ type: 'segment-started', pageCount: pages.length })
    try {
      await Promise.all(pages.map(pageIndex => this.options.scheduler.pages.run(async () => {
        try {
          const page = await new PageDataflow({ ...this.options, pageIndex, signal, emit: event => this.handlePageEvent(event) }).run()
          this.latestPages.set(pageIndex, page)
        } catch (error) {
          this.emit({ type: 'page-failed', pageIndex, error, metrics: {} })
          if (this.options.stopOnError) throw error
        }
      })))
      const donePages = pages.map(pageIndex => this.latestPages.get(pageIndex)).filter((page): page is RenderedPage => !!page)
      this.emit({ type: 'segment-done', pages: donePages })
      return donePages
    } finally {
      this.eventBuffer.close()
    }
  }

  private handlePageEvent(event: SegmentEvent): void {
    if (event.type === 'page-display-ready') this.latestPages.set(event.pageIndex, event.page)
    this.emit(event)
  }

  private combinedSignal(): AbortSignal {
    if (this.options.signal?.aborted) this.abortController.abort(this.options.signal.reason)
    else this.options.signal?.addEventListener('abort', () => this.abortController.abort(this.options.signal?.reason), { once: true })
    return this.abortController.signal
  }

  private emit(event: SegmentEvent): void {
    this.eventBuffer.emit(event)
  }
}

function range(length: number): number[] {
  return Array.from({ length }, (_, index) => index)
}
