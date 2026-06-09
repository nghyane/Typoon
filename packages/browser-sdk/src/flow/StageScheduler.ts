import { AsyncLimiter } from './AsyncLimiter'

export interface StageConcurrencyPolicy {
  readonly pages: number
  readonly recognize: number
  readonly detect: number
  readonly translate: number
}

export interface StageRetryOptions {
  readonly attempts?: number
}

export interface StageSchedulerPolicy {
  readonly concurrency: StageConcurrencyPolicy
  readonly retry?: {
    readonly recognize?: StageRetryOptions
    readonly detect?: StageRetryOptions
    readonly translate?: StageRetryOptions
  }
}

export type StageSchedulerOptions = Partial<StageSchedulerPolicy> & {
  readonly concurrency?: Partial<StageConcurrencyPolicy>
}

export class StageScheduler {
  readonly pages: AsyncLimiter
  private readonly recognizeLimiter: AsyncLimiter
  private readonly detectLimiter: AsyncLimiter
  private readonly translateLimiter: AsyncLimiter

  constructor(private readonly policy: StageSchedulerPolicy) {
    this.pages = new AsyncLimiter(Math.max(1, policy.concurrency.pages))
    this.recognizeLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.recognize))
    this.detectLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.detect))
    this.translateLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.translate))
  }

  recognize<T>(task: () => Promise<T>): Promise<T> {
    return this.recognizeLimiter.run(() => retry(task, this.policy.retry?.recognize?.attempts ?? 1))
  }

  detect<T>(task: () => Promise<T>): Promise<T> {
    return this.detectLimiter.run(() => retry(task, this.policy.retry?.detect?.attempts ?? 1))
  }

  translate<T>(task: () => Promise<T>): Promise<T> {
    return this.translateLimiter.run(() => retry(task, this.policy.retry?.translate?.attempts ?? 1))
  }
}

async function retry<T>(task: () => Promise<T>, attempts: number): Promise<T> {
  let lastError: unknown
  for (let attempt = 0; attempt < Math.max(1, attempts); attempt++) {
    try {
      return await task()
    } catch (error) {
      lastError = error
    }
  }
  throw lastError
}
