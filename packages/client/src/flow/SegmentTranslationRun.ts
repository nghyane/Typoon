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

import { canvasPageFromImage } from '../pipeline/canvasPageFromImage'
import { textUnitsFromBlocks } from '../pipeline/textUnits'
import { translationUnitsFromTextUnits } from '../pipeline/translationUnits'
import { textPlacementsFromRecognition, layoutPlacementsFromRegions } from '../pipeline/textPlacements'
import { translateSegment } from '../pipeline/translateSegment'
import { materializeRenderedPage } from '../pipeline/materializeRenderedPage'

type PageListener = (page: RenderedPage) => void

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
  private readonly listeners = new Set<PageListener>()
  private readonly abort = new AbortController()
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
    private readonly request: SegmentRequest,
    private readonly deps: {
      readonly source: PageSource
      readonly recognizer: TextRecognizer
      readonly translator: Translator
      readonly postEditor?: TranslationPostEditor
      readonly detector?: TextRegionDetector
      readonly scheduler: StageScheduler
      readonly signal?: AbortSignal
    },
  ) {
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

  // -- graph --

  private async run(): Promise<void> {
    const signal = this.combinedSignal()

    const frames = await this.pages(signal)
    this.work.frames = frames

    const regionsTask = this.regions(frames, signal)

    const transcript = await this.transcript(frames, signal)
    this.work.transcript = transcript

    const script = this.script(transcript)
    this.work.script = script
    this.work.unitsByPage = groupByPage(script)

    this.useLayout(this.planLayout())

    const machine = await this.machine(script, signal)
    this.work.machineVersion = machine
    this.emitPages()

    if (this.request.postEdit && this.deps.postEditor && this.work.layout) {
      const edited = await this.postEdit({
        transcript,
        script,
        base: machine,
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

  // -- graph nodes --

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

  private async transcript(
    frames: CanvasPage[],
    signal: AbortSignal,
  ): Promise<RecognizedTextPage[]> {
    const pages = await Promise.all(
      frames.map(frame =>
        this.deps.scheduler.recognize(() =>
          this.deps.recognizer.recognizeText(frame.image, {
            pageIndex: frame.pageIndex,
            sourceLang: this.request.sourceLang,
            signal,
          }),
        ),
      ),
    )
    return pages.sort((a, b) => a.pageIndex - b.pageIndex)
  }

  private async regions(
    frames: CanvasPage[],
    signal: AbortSignal,
  ): Promise<(readonly TextRegion[])[] | null> {
    const detector = this.deps.detector
    if (!detector) return null

    return Promise.all(
      frames.map(frame =>
        this.deps.scheduler.detect(async () =>
          detector.detectTextRegions(frame.image, { signal }),
        ),
      ),
    )
  }

  private script(transcript: RecognizedTextPage[]): TranslationUnit[] {
    return transcript.flatMap(page => {
      const textUnits = textUnitsFromBlocks(page.blocks, page.pageIndex)
      return translationUnitsFromTextUnits(textUnits)
    })
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

  private async machine(
    script: TranslationUnit[],
    signal: AbortSignal,
  ): Promise<TranslationVersion> {
    const units = await translateSegment({
      units: script,
      translator: this.deps.translator,
      sourceLang: this.request.sourceLang,
      targetLang: this.request.targetLang,
      signal,
    })

    return { id: this.nextVersionId(), method: 'machine', units }
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

  // -- state + display --

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

function groupTranslatedByPage(units: readonly TranslatedUnit[]): Map<number, TranslatedUnit[]> {
  const byPage = new Map<number, TranslatedUnit[]>()

  for (const unit of units) {
    const list = byPage.get(unit.pageIndex)
    if (list) list.push(unit)
    else byPage.set(unit.pageIndex, [unit])
  }

  return byPage
}
