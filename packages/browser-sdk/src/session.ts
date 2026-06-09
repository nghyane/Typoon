import { BrowserSdkError } from './errors'
import type { ContextBuilder } from './context/ContextBuilder'
import type { TextRegionDetector } from './detectors/textRegions'
import { isCapability, type Capability, type StatusListener, type StatusSnapshot, type Unsubscribe } from './domain/capability'
import type { TranslatedPage } from './domain/translation'
import type { ImageInput } from './image/input'
import { materializePagePlan, translatedPageFromMaterializedPlan, type MaterializedPagePlan } from './pipeline/materializePage'
import { translateSegment } from './pipeline/translateSegment'
import type { TextRecognizer } from './recognizers/text'
import type { Translator } from './translators/translator'

export type LayoutHint = 'auto' | 'paged' | 'vertical-scroll'
export type ContinuityHint = 'unknown' | 'discrete' | 'continuous'
export type SegmentKind = 'chapter' | 'range' | 'upload'
export type ArtifactName = 'translated'

export interface ResourceBudgets {
  readonly recognizeConcurrency?: number
  readonly translateConcurrency?: number
}

export interface TranslationSessionOptions {
  readonly sourceLang?: string
  readonly targetLang?: string
  readonly recognizer: TextRecognizer
  readonly detector?: TextRegionDetector
  readonly translator: Translator
  readonly contextBuilder?: ContextBuilder
  readonly budgets?: ResourceBudgets
}

export interface WorkOptions {
  readonly id: string
  readonly sourceLang?: string
  readonly targetLang?: string
  readonly layoutHint?: LayoutHint
  readonly continuityHint?: ContinuityHint
}

export interface PageSource {
  readonly pageCount: number
  loadPage(index: number, signal?: AbortSignal): ImageInput | Promise<ImageInput>
}

export interface SegmentOptions {
  readonly id: string
  readonly kind?: SegmentKind
  readonly pages: PageSource
  readonly layoutHint?: LayoutHint
  readonly continuity?: ContinuityHint
}

export interface MaterializeOptions {
  readonly signal?: AbortSignal
}

export interface EnsureOptions extends MaterializeOptions {
  readonly artifacts: readonly ArtifactName[]
  readonly pages?: readonly number[]
}

export class TranslationSession {
  readonly budgets: ResourceBudgets
  private readonly statusListeners = new Set<StatusListener<StatusSnapshot>>()
  private readonly capabilities: readonly Capability[]
  private readonly capabilityUnsubscribes: readonly Unsubscribe[]

  constructor(private readonly options: TranslationSessionOptions) {
    this.budgets = options.budgets ?? {}
    const candidates: readonly unknown[] = [options.recognizer, options.detector, options.translator, options.contextBuilder]
    this.capabilities = candidates.filter(isCapability)
    this.capabilityUnsubscribes = this.capabilities.map(capability => capability.subscribeStatus(() => this.emitStatus()))
  }

  openWork(options: WorkOptions): Work {
    return new Work(this.options, options)
  }

  async ensureReady(options: { readonly signal?: AbortSignal } = {}): Promise<void> {
    await Promise.all(this.capabilities.map(capability => capability.ensureReady(options)))
  }

  statusSnapshot(): StatusSnapshot {
    return { capabilities: this.capabilities.map(capability => capability.status()) }
  }

  subscribeStatus(listener: StatusListener<StatusSnapshot>): Unsubscribe {
    this.statusListeners.add(listener)
    listener(this.statusSnapshot())
    return () => this.statusListeners.delete(listener)
  }

  close(): void {
    for (const unsubscribe of this.capabilityUnsubscribes) unsubscribe()
    this.statusListeners.clear()
  }

  private emitStatus(): void {
    const snapshot = this.statusSnapshot()
    for (const listener of this.statusListeners) listener(snapshot)
  }
}

export class Work {
  readonly id: string
  readonly sourceLang: string | null
  readonly targetLang: string
  readonly layoutHint: LayoutHint
  readonly continuityHint: ContinuityHint

  constructor(
    private readonly sessionOptions: TranslationSessionOptions,
    options: WorkOptions,
  ) {
    this.id = options.id
    this.sourceLang = options.sourceLang ?? sessionOptions.sourceLang ?? null
    this.targetLang = resolveTargetLang(sessionOptions, options)
    this.layoutHint = options.layoutHint ?? 'auto'
    this.continuityHint = options.continuityHint ?? 'unknown'
  }

  openSegment(options: SegmentOptions): Segment {
    return new Segment(this.sessionOptions, this, options)
  }
}

export class Segment {
  readonly id: string
  readonly kind: SegmentKind
  readonly pageCount: number
  readonly layoutHint: LayoutHint
  readonly continuityHint: ContinuityHint
  private readonly translatedPages = new Map<number, Promise<TranslatedPage>>()
  private readonly pagePlans = new Map<number, Promise<MaterializedPagePlan>>()

  constructor(
    private readonly sessionOptions: TranslationSessionOptions,
    private readonly work: Work,
    private readonly options: SegmentOptions,
  ) {
    this.id = options.id
    this.kind = options.kind ?? 'chapter'
    this.pageCount = options.pages.pageCount
    this.layoutHint = options.layoutHint ?? work.layoutHint
    this.continuityHint = options.continuity ?? work.continuityHint
  }

  page(index: number): PageHandle {
    assertPageIndex(index, this.pageCount)
    return new PageHandle(index, this)
  }

  async ensure(options: EnsureOptions): Promise<readonly TranslatedPage[]> {
    if (!options.artifacts.includes('translated')) return []
    const pages = options.pages ?? range(this.pageCount)
    if (pages.length <= 1) {
      const concurrency = Math.max(1, this.sessionOptions.budgets?.translateConcurrency ?? 1)
      return mapWithConcurrency(pages, concurrency, index => this.translatePage(index, options))
    }

    const missingPages = pages.filter(index => !this.translatedPages.has(index))
    const batchResults = missingPages.length ? await this.translatePagesAsBatch(missingPages, options) : new Map<number, TranslatedPage>()
    return Promise.all(pages.map(index => batchResults.get(index) ?? this.translatePage(index, options)))
  }

  planPage(index: number, options: MaterializeOptions = {}): Promise<MaterializedPagePlan> {
    assertPageIndex(index, this.pageCount)
    if (options.signal) return this.materializePlan(index, options)
    const cached = this.pagePlans.get(index)
    if (cached) return cached
    const promise = this.materializePlan(index, options)
    this.pagePlans.set(index, promise)
    promise.catch(() => this.pagePlans.delete(index))
    return promise
  }

  translatePage(index: number, options: MaterializeOptions = {}): Promise<TranslatedPage> {
    assertPageIndex(index, this.pageCount)
    if (options.signal) return this.materializeTranslatedPage(index, options)
    const cached = this.translatedPages.get(index)
    if (cached) return cached
    const promise = this.materializeTranslatedPage(index, options)
    this.translatedPages.set(index, promise)
    promise.catch(() => this.translatedPages.delete(index))
    return promise
  }

  private async materializePlan(index: number, options: MaterializeOptions): Promise<MaterializedPagePlan> {
    const input = await this.options.pages.loadPage(index, options.signal)
    throwIfAborted(options.signal)
    return materializePagePlan(input, {
      recognizer: this.sessionOptions.recognizer,
      detector: this.sessionOptions.detector,
      sourceLang: this.work.sourceLang,
      pageIndex: index,
      signal: options.signal,
    })
  }

  private async materializeTranslatedPage(index: number, options: MaterializeOptions): Promise<TranslatedPage> {
    const plan = await this.planPage(index, options)
    const context = this.sessionOptions.contextBuilder
      ? await this.sessionOptions.contextBuilder.buildContext({
        workId: this.work.id,
        segmentId: this.id,
        sourceLang: this.work.sourceLang,
        targetLang: this.work.targetLang,
        units: plan.plan.units,
        signal: options.signal,
      })
      : undefined
    const translations = await translateSegment({
      units: plan.plan.units,
      translator: this.sessionOptions.translator,
      sourceLang: this.work.sourceLang,
      targetLang: this.work.targetLang,
      context,
      signal: options.signal,
    })
    throwIfAborted(options.signal)
    return translatedPageFromMaterializedPlan(plan, translations)
  }

  private async translatePagesAsBatch(pages: readonly number[], options: MaterializeOptions): Promise<Map<number, TranslatedPage>> {
    const concurrency = Math.max(1, this.sessionOptions.budgets?.recognizeConcurrency ?? 1)
    const pageArtifacts = await mapWithConcurrency(pages, concurrency, index => this.planPage(index, options))
    const units = pageArtifacts.flatMap(page => page.plan.units)
    const context = this.sessionOptions.contextBuilder
      ? await this.sessionOptions.contextBuilder.buildContext({
        workId: this.work.id,
        segmentId: this.id,
        sourceLang: this.work.sourceLang,
        targetLang: this.work.targetLang,
        units,
        signal: options.signal,
      })
      : undefined
    const translations = await translateSegment({
      units,
      translator: this.sessionOptions.translator,
      sourceLang: this.work.sourceLang,
      targetLang: this.work.targetLang,
      context,
      signal: options.signal,
    })
    throwIfAborted(options.signal)

    const results = new Map<number, TranslatedPage>()
    for (const artifact of pageArtifacts) {
      const pageTranslations = translations.filter(unit => unit.pageIndex === artifact.canvas.pageIndex)
      const result = translatedPageFromMaterializedPlan(artifact, pageTranslations)
      results.set(artifact.canvas.pageIndex, result)
      if (!options.signal) this.translatedPages.set(artifact.canvas.pageIndex, Promise.resolve(result))
    }
    return results
  }
}

export class PageHandle {
  constructor(
    readonly index: number,
    private readonly segment: Segment,
  ) {}

  plan(options?: MaterializeOptions): Promise<MaterializedPagePlan> {
    return this.segment.planPage(this.index, options)
  }

  translate(options?: MaterializeOptions): Promise<TranslatedPage> {
    return this.segment.translatePage(this.index, options)
  }
}

function resolveTargetLang(sessionOptions: TranslationSessionOptions, workOptions: WorkOptions): string {
  const targetLang = workOptions.targetLang ?? sessionOptions.targetLang
  if (!targetLang) throw new BrowserSdkError('MODEL_UNAVAILABLE', 'targetLang is required for translation work')
  return targetLang
}

function assertPageIndex(index: number, pageCount: number): void {
  if (!Number.isInteger(index) || index < 0 || index >= pageCount) {
    throw new RangeError(`page index out of range: ${index}`)
  }
}

function range(length: number): number[] {
  return Array.from({ length }, (_, index) => index)
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

async function mapWithConcurrency<T, R>(items: readonly T[], concurrency: number, mapper: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length)
  let nextIndex = 0

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex
      nextIndex += 1
      const item = items[index]
      if (item === undefined) continue
      results[index] = await mapper(item)
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()))
  return results
}
