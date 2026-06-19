import { AsyncLimiter } from './AsyncLimiter'

export interface StageConcurrencyPolicy {
  readonly pages: number
  readonly recognize: number
  readonly detect: number
  readonly translate: number
}

export interface StageRetryOptions {
  readonly attempts?: number
  readonly baseDelayMs?: number
  readonly maxDelayMs?: number
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
  private readonly policy: StageSchedulerPolicy
  private readonly recognizeLimiter: AsyncLimiter
  private readonly detectLimiter: AsyncLimiter
  private readonly translateLimiter: AsyncLimiter

  constructor(policy: StageSchedulerPolicy) {
    this.policy = policy
    this.pages = new AsyncLimiter(Math.max(1, policy.concurrency.pages))
    this.recognizeLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.recognize))
    this.detectLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.detect))
    this.translateLimiter = new AsyncLimiter(Math.max(1, policy.concurrency.translate))
  }

  recognize<T>(task: () => Promise<T>): Promise<T> {
    return this.recognizeLimiter.run(() => retry(task, this.policy.retry?.recognize))
  }

  detect<T>(task: () => Promise<T>): Promise<T> {
    return this.detectLimiter.run(() => retry(task, this.policy.retry?.detect))
  }

  translate<T>(task: () => Promise<T>): Promise<T> {
    return this.translateLimiter.run(() => retry(task, this.policy.retry?.translate))
  }
}

async function retry<T>(task: () => Promise<T>, options: StageRetryOptions | undefined): Promise<T> {
  const attempts = Math.max(1, options?.attempts ?? 1)
  let lastError: unknown
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await task()
    } catch (error) {
      lastError = error
      if (attempt < attempts - 1) await sleep(retryDelayMs(attempt, options))
    }
  }
  throw lastError
}

function retryDelayMs(attempt: number, options: StageRetryOptions | undefined): number {
  const base = options?.baseDelayMs ?? 0
  if (base <= 0) return 0
  const max = Math.max(base, options?.maxDelayMs ?? base)
  const exponential = Math.min(max, base * 2 ** attempt)
  return exponential + Math.round(Math.random() * exponential * 0.3)
}

function sleep(ms: number): Promise<void> {
  return ms <= 0 ? Promise.resolve() : new Promise(resolve => setTimeout(resolve, ms))
}
