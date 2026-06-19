import { AsyncLimiter } from '../flow/AsyncLimiter'

export interface PipelineConcurrency {
  /** Max full-page pixel pipelines allowed to be live at once. */
  readonly maxPreparedPages?: number
  readonly load: number
  readonly prepare: number
  readonly ocr: number
  readonly detect: number
  readonly translate: number
  readonly compose: number
}

/**
 * Concurrency + retry gate for each pipeline stage.
 *
 * Owns no page state, no progress, no queue awareness.
 * Just "how many of this stage can run at once, and should I retry?"
 */
export class StageExecutor {
  private readonly loadLimit: AsyncLimiter
  private readonly prepareLimit: AsyncLimiter
  private readonly ocrLimit: AsyncLimiter
  private readonly detectLimit: AsyncLimiter
  private readonly translateLimit: AsyncLimiter
  private readonly composeLimit: AsyncLimiter

  constructor(concurrency: PipelineConcurrency) {
    this.loadLimit = new AsyncLimiter(Math.max(1, concurrency.load))
    this.prepareLimit = new AsyncLimiter(Math.max(1, concurrency.prepare))
    this.ocrLimit = new AsyncLimiter(Math.max(1, concurrency.ocr))
    this.detectLimit = new AsyncLimiter(Math.max(1, concurrency.detect))
    this.translateLimit = new AsyncLimiter(Math.max(1, concurrency.translate))
    this.composeLimit = new AsyncLimiter(Math.max(1, concurrency.compose))
  }

  load<T>(task: () => Promise<T>): Promise<T> {
    return this.loadLimit.run(task)
  }

  prepare<T>(task: () => Promise<T>): Promise<T> {
    return this.prepareLimit.run(task)
  }

  ocr<T>(task: () => Promise<T>): Promise<T> {
    return this.ocrLimit.run(task)
  }

  detect<T>(task: () => Promise<T>): Promise<T> {
    return this.detectLimit.run(task)
  }

  translate<T>(task: () => Promise<T>): Promise<T> {
    return this.translateLimit.run(task)
  }

  compose<T>(task: () => Promise<T>): Promise<T> {
    return this.composeLimit.run(task)
  }
}
