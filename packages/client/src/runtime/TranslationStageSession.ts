import type { PageOverlay } from '../domain/overlay'
import type { PreparedPageHandle } from '../domain/prepared'
import type { TranslationRequest } from '../domain/run'
import type { PageAsset } from '../domain/source'
import type { RecognizedTextPage } from '../domain/text'
import type { TextRegion } from '../domain/regions'
import type { VisionRuntime, PreparationSession } from '../vision/VisionRuntime'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import { preparedTextFromRecognition, translatePreparedText, type PreparedTextResult, type PreparedTranslationResult } from '../pipeline/translatePreparedPage'
import { buildOverlayPlacements, composeAndProjectOverlays } from '../pipeline/composeOverlay'

export interface TranslationStageSessionDeps {
  readonly vision: VisionRuntime
  readonly recognizer: TextRecognizer
  readonly translator: Translator
}

export class TranslationStageSession {
  readonly runId: string
  private readonly deps: TranslationStageSessionDeps
  private readonly session: PreparationSession
  private disposed = false

  constructor(
    runId: string,
    session: PreparationSession,
    deps: TranslationStageSessionDeps,
  ) {
    this.runId = runId
    this.session = session
    this.deps = deps
  }

  preparePage(asset: PageAsset, signal?: AbortSignal): Promise<readonly PreparedPageHandle[]> {
    this.assertActive()
    return this.session.push(asset, signal)
  }

  flush(signal?: AbortSignal): Promise<readonly PreparedPageHandle[]> {
    this.assertActive()
    return this.session.flush(signal)
  }

  async createSeamRepair(
    top: PreparedPageHandle,
    bottom: PreparedPageHandle,
    bandPx = 320,
    signal?: AbortSignal,
  ): Promise<PreparedPageHandle | null> {
    this.assertActive()
    if (!this.deps.vision.createSeamRepair) throw new Error('vision runtime does not support seam repair')
    return this.deps.vision.createSeamRepair(top, bottom, bandPx, signal)
  }

  async recognize(
    handle: PreparedPageHandle,
    sourceLanguage: string | null,
    signal?: AbortSignal,
  ): Promise<PreparedTextResult> {
    this.assertActive()
    const options = { pageIndex: handle.preparedPageIndex, sourceLang: sourceLanguage, signal }
    const encoded = this.deps.recognizer.recognizeEncoded
      ? await this.deps.vision.encodeForOcr(handle, signal)
      : null
    const recognized: RecognizedTextPage = encoded
      ? await this.deps.recognizer.recognizeEncoded!(encoded, options)
      : await this.deps.recognizer.recognizeText(
          await this.deps.vision.readPixels(handle, signal),
          options,
        )
    return preparedTextFromRecognition({ handle, recognized })
  }

  translate(
    text: PreparedTextResult,
    sourceLanguage: string | null,
    targetLanguage: string,
    signal?: AbortSignal,
  ): Promise<PreparedTranslationResult> {
    this.assertActive()
    return translatePreparedText({
      text,
      translator: this.deps.translator,
      sourceLanguage,
      targetLanguage,
      signal,
    })
  }

  async composeOverlay(
    handle: PreparedPageHandle,
    translated: PreparedTranslationResult,
    regions: readonly TextRegion[] | null = null,
    signal?: AbortSignal,
  ): Promise<readonly PageOverlay[]> {
    this.assertActive()
    const placements = buildOverlayPlacements({
      recognized: translated.recognized,
      textUnits: translated.textUnits,
      regions,
    })
    const placementMargins = await this.deps.vision.estimateMargins(handle, placements, signal)
    return composeAndProjectOverlays({
      handle,
      recognized: translated.recognized,
      textUnits: translated.textUnits,
      translations: translated.translations,
      regions,
      placements,
      placementMargins,
    })
  }

  async recognizeTranslateCompose(args: {
    readonly handle: PreparedPageHandle
    readonly sourceLanguage: string | null
    readonly targetLanguage: string
    readonly regions?: readonly TextRegion[] | null
    readonly signal?: AbortSignal
  }): Promise<readonly PageOverlay[]> {
    const text = await this.recognize(args.handle, args.sourceLanguage, args.signal)
    const translated = await this.translate(text, args.sourceLanguage, args.targetLanguage, args.signal)
    return this.composeOverlay(args.handle, translated, args.regions ?? null, args.signal)
  }

  release(handle: PreparedPageHandle): void {
    this.deps.vision.release(handle)
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.session.dispose?.()
  }

  private assertActive(): void {
    if (this.disposed) throw new Error('translation stage session is disposed')
  }
}

export type RuntimeStageRequest = Pick<TranslationRequest, 'preparation'>
