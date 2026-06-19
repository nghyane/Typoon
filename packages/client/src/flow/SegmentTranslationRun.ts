import type { CanvasPage } from '../domain/canvas'
import type { PageSource } from '../domain/source'
import type { RecognizedTextPage } from '../domain/text'
import type { TextRegion } from '../domain/regions'
import type { RenderedPage, TranslatedUnit, TranslationUnit } from '../domain/translation'
import type { TextRecognizer } from '../recognizers/text'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { Translator } from '../translators/translator'
import type { TranslationPostEditor } from '../translators/postEditor'
import type { StageScheduler } from './StageScheduler'
import type { LayoutPlan, SegmentRequest, TranslationVersion } from '../domain/segment'
import type { SegmentDisplayOptions } from './TranslationEngine'

import { canvasPageFromImage } from '../pipeline/canvasPageFromImage'
import { textUnitsFromBlocks } from '../pipeline/textUnits'
import { translationUnitsFromTextUnits } from '../pipeline/translationUnits'
import { textPlacementsFromRecognition, layoutPlacementsFromRegions } from '../pipeline/textPlacements'
import { materializeRenderedPage } from '../pipeline/materializeRenderedPage'

type PageListener = (page: RenderedPage) => void

type RecognizedPageWork = {
  readonly page: RecognizedTextPage
  readonly pageUnits: readonly TranslationUnit[]
}

type PageState = {
  readonly frame: CanvasPage
  recognized?: RecognizedTextPage
  pageUnits?: readonly TranslationUnit[]
  translations?: readonly TranslatedUnit[]
  regions?: readonly TextRegion[]
}

type Work = {
  frames: CanvasPage[] | null
  transcript: RecognizedTextPage[] | null
  regions: (readonly TextRegion[])[] | null
  script: TranslationUnit[] | null
  unitsByPage: Map<number, TranslationUnit[]> | null
  layout: LayoutPlan | null

  machineVersion: TranslationVersion | null
  postEditVersion: TranslationVersion | null

  versionSeq: number
  pages: RenderedPage[]
}

export class SegmentTranslationRun {
  private readonly request: SegmentRequest
  private readonly deps: {
    readonly source: PageSource
    readonly recognizer: TextRecognizer
    readonly translator: Translator
    readonly postEditor?: TranslationPostEditor
    readonly detector?: TextRegionDetector
    readonly scheduler: StageScheduler
    readonly display?: SegmentDisplayOptions
    readonly signal?: AbortSignal
  }
  private readonly listeners = new Set<PageListener>()
  private readonly abort = new AbortController()
  private readonly pageStates = new Map<number, PageState>()
  private started = false
  private cancelled = false
  private readonly work: Work = {
    frames: null,
    transcript: null,
    regions: null,
    script: null,
    layout: null,
    unitsByPage: null,
    machineVersion: null,
    postEditVersion: null,
    versionSeq: 0,
    pages: [],
  }

  private doneResolve!: (pages: readonly RenderedPage[]) => void
  private doneReject!: (error: unknown) => void
  readonly done: Promise<readonly RenderedPage[]>

  constructor(
    request: SegmentRequest,
    deps: {
      readonly source: PageSource
      readonly recognizer: TextRecognizer
      readonly translator: Translator
      readonly postEditor?: TranslationPostEditor
      readonly detector?: TextRegionDetector
      readonly scheduler: StageScheduler
      readonly display?: SegmentDisplayOptions
      readonly signal?: AbortSignal
    },
  ) {
    this.request = request
    this.deps = deps
    this.done = new Promise((resolve, reject) => {
      this.doneResolve = resolve
      this.doneReject = reject
    })
  }

  onDisplay(fn: PageListener): () => void {
    this.listeners.add(fn)
    return () => { this.listeners.delete(fn) }
  }

  cancel(reason?: Error): void {
    this.cancelled = true
    if (!this.abort.signal.aborted) this.abort.abort(reason)
    this.doneResolve(this.work.pages)
  }

  start(): void {
    if (this.started) return
    this.started = true

    this.run().catch(error => {
      this.doneReject(error instanceof Error ? error : new Error(String(error)))
    })
  }

  private async run(): Promise<void> {
    const signal = this.combinedSignal()

    const frames = await this.pages(signal)
    this.work.frames = frames

    for (const frame of frames) {
      this.pageStates.set(frame.pageIndex, { frame })
    }

    const translationTasks: Array<Promise<readonly TranslatedUnit[]>> = []

    const regionsTask = this.regions(frames, signal)

    const recognized: RecognizedPageWork[] = await Promise.all(frames.map(frame =>
      this.deps.scheduler.recognize(async () => {
        const page = await this.deps.recognizer.recognizeText(frame.image, {
          pageIndex: frame.pageIndex,
          sourceLang: this.request.sourceLang,
          signal,
        })

        const textUnits = textUnitsFromBlocks(page.blocks, page.pageIndex)
        const pageUnits = translationUnitsFromTextUnits(textUnits)
        const state = this.pageStates.get(frame.pageIndex)!

        state.recognized = page
        state.pageUnits = pageUnits

        const progressive = this.deps.display?.progressive
        if (progressive) {
          const task = this.translateAndEmit(state, signal)
          void task.catch(() => {})
          translationTasks.push(task)
        }

        return { page, pageUnits }
      }),
    ))

    recognized.sort((a, b) => a.page.pageIndex - b.page.pageIndex)

    const transcript = recognized.map(item => item.page)
    const script = recognized.flatMap(item => item.pageUnits)

    let translatedUnits: readonly TranslatedUnit[]

    if (!this.deps.display?.progressive) {
      // Batch mode: translate full script
      translatedUnits = await this.deps.translator.translateUnits({
        units: script,
        sourceLang: this.request.sourceLang ?? null,
        targetLang: this.request.targetLang,
        signal,
      })
      for (const [pageIndex, state] of this.pageStates) {
        state.translations = translatedUnits.filter(u => u.pageIndex === pageIndex)
      }
    } else {
      // Progressive mode: wait all page tasks
      const outcomes = await Promise.allSettled(translationTasks)
      const errors = outcomes
        .filter((o): o is PromiseRejectedResult => o.status === 'rejected')
        .map(o => o.reason)
      if (errors.length > 0) throw errors[0]!
      translatedUnits = outcomes
        .filter((o): o is PromiseFulfilledResult<readonly TranslatedUnit[]> => o.status === 'fulfilled')
        .flatMap(o => o.value)
    }

    translatedUnits = orderTranslatedUnits(translatedUnits, script)

    this.work.transcript = transcript
    this.work.script = script
    this.work.unitsByPage = groupByPage(script)
    this.work.machineVersion = { id: this.nextVersionId(), method: 'machine', units: translatedUnits }

    this.useLayout(this.planLayout())

    if (!this.deps.display?.progressive) {
      this.emitPages()
    }

    if (this.request.postEdit && this.deps.postEditor && this.work.layout) {
      const edited = await this.postEdit({
        transcript,
        script,
        base: this.work.machineVersion,
        layout: this.work.layout,
        signal,
      })
      this.work.postEditVersion = edited
      this.emitPages()
    }

    const regions = await regionsTask.catch(() => null)
    if (regions) {
      this.work.regions = regions
      this.useLayout(this.planLayout())
      this.emitPages()
    }

    this.doneResolve(this.work.pages)
  }

  private async translateAndEmit(state: PageState, signal: AbortSignal): Promise<readonly TranslatedUnit[]> {
    const translations = await this.deps.translator.translateUnits({
      units: state.pageUnits!,
      sourceLang: this.request.sourceLang ?? null,
      targetLang: this.request.targetLang,
      signal,
    })
    state.translations = translations
    this.tryEmitPage(state.frame.pageIndex)
    return translations
  }

  private tryEmitPage(pageIndex: number): void {
    if (this.cancelled) return

    const state = this.pageStates.get(pageIndex)
    if (!state?.recognized || !state.pageUnits || !state.translations) return

    const textUnits = textUnitsFromBlocks(state.recognized.blocks, state.recognized.pageIndex)
    const placements = state.regions?.length
      ? layoutPlacementsFromRegions(state.recognized, textUnits, state.regions)
      : textPlacementsFromRecognition(state.recognized, textUnits)

    const rendered = materializeRenderedPage({
      phase: 'text',
      canvas: state.frame,
      recognizedText: state.recognized,
      textUnits,
      translationUnits: state.pageUnits,
      placements,
      translations: state.translations,
    })

    const existing = this.work.pages.findIndex(page => page.pageIndex === rendered.pageIndex)
    if (existing === -1) this.work.pages.push(rendered)
    else this.work.pages[existing] = rendered

    this.work.pages.sort((a, b) => a.pageIndex - b.pageIndex)

    for (const fn of this.listeners) {
      fn(rendered)
    }
  }

  private combinedSignal(): AbortSignal {
    if (this.deps.signal?.aborted) {
      this.abort.abort(this.deps.signal.reason)
    } else {
      this.deps.signal?.addEventListener(
        'abort',
        () => { this.abort.abort(this.deps.signal?.reason) },
        { once: true },
      )
    }
    return this.abort.signal
  }

  private async pages(signal: AbortSignal): Promise<CanvasPage[]> {
    const frames = await Promise.all(
      this.request.pageIndexes.map(pageIndex =>
        this.deps.scheduler.pages.run(async () => {
          const input = await this.deps.source.loadPage(pageIndex, signal)
          return canvasPageFromImage(input, { pageIndex })
        }),
      ),
    )
    return frames.sort((a, b) => a.pageIndex - b.pageIndex)
  }

  private async regions(
    frames: CanvasPage[],
    signal: AbortSignal,
  ): Promise<(readonly TextRegion[])[] | null> {
    const detector = this.deps.detector
    if (!detector) return null

    return Promise.all(
      frames.map(frame =>
        this.deps.scheduler.detect(async () => {
          const regions = await detector.detectTextRegions(frame.image, { signal })
          const state = this.pageStates.get(frame.pageIndex)
          if (state) state.regions = regions
          return regions
        }),
      ),
    )
  }

  private planLayout(): LayoutPlan {
    const { transcript, script, regions } = this.work
    if (!transcript || !script) throw new Error('cannot plan layout without transcript + script')

    return {
      pages: transcript.map((page, i) => {
        const textUnits = textUnitsFromBlocks(page.blocks, page.pageIndex)
        const regionSet = regions?.[i]

        return {
          pageIndex: page.pageIndex,
          placements: regionSet
            ? layoutPlacementsFromRegions(page, textUnits, regionSet)
            : textPlacementsFromRecognition(page, textUnits),
        }
      }),
    }
  }

  private async postEdit(args: {
    transcript: RecognizedTextPage[]
    script: TranslationUnit[]
    base: TranslationVersion
    layout: LayoutPlan
    signal: AbortSignal
  }): Promise<TranslationVersion> {
    const editor = this.deps.postEditor!
    const version = await editor.postEdit({
      request: this.request,
      transcript: args.transcript,
      script: { units: args.script },
      base: args.base,
      layout: args.layout,
      signal: args.signal,
    })

    return {
      ...version,
      method: 'post_edit',
      baseId: version.baseId ?? args.base.id,
    }
  }

  private useLayout(plan: LayoutPlan): void {
    this.work.layout = plan
  }

  private emitPages(): void {
    if (this.cancelled) return

    const { frames, transcript, unitsByPage, layout } = this.work
    const translation = this.work.postEditVersion ?? this.work.machineVersion
    if (!frames || !transcript || !unitsByPage || !layout || !translation) return

    const translatedByPage = groupTranslatedByPage(translation.units)

    this.work.pages = []

    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i]!
      const page = transcript[i]!
      const pageUnits = unitsByPage.get(frame.pageIndex) ?? []
      const layoutPage = layout.pages[i]!
      const pageTranslations = translatedByPage.get(frame.pageIndex) ?? []
      const textUnits = textUnitsFromBlocks(page.blocks, page.pageIndex)

      const rendered = materializeRenderedPage({
        phase: 'text',
        canvas: frame,
        recognizedText: page,
        textUnits,
        translationUnits: pageUnits,
        placements: layoutPage.placements,
        translations: pageTranslations,
      })

      this.work.pages.push(rendered)

      for (const fn of this.listeners) {
        fn(rendered)
      }
    }
  }

  private nextVersionId(): string {
    return `v${++this.work.versionSeq}`
  }
}

function groupByPage(units: readonly TranslationUnit[]): Map<number, TranslationUnit[]> {
  const byPage = new Map<number, TranslationUnit[]>()

  for (const unit of units) {
    const list = byPage.get(unit.pageIndex)
    if (list) list.push(unit)
    else byPage.set(unit.pageIndex, [unit])
  }

  return byPage
}

function orderTranslatedUnits(
  units: readonly TranslatedUnit[],
  script: readonly TranslationUnit[],
): TranslatedUnit[] {
  const order = new Map(script.map((unit, index) => [unit.id, index]))
  return [...units].sort((a, b) => (order.get(a.unitId) ?? Number.MAX_SAFE_INTEGER) - (order.get(b.unitId) ?? Number.MAX_SAFE_INTEGER))
}

function groupTranslatedByPage(units: readonly TranslatedUnit[]): Map<number, TranslatedUnit[]> {
  const byPage = new Map<number, TranslatedUnit[]>()

  for (const unit of units) {
    const list = byPage.get(unit.pageIndex)
    if (list) list.push(unit)
    else byPage.set(unit.pageIndex, [unit])
  }

  return byPage
}
