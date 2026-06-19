/**
 * Vision runtime that runs everything on the main thread.
 *
 * Used as fallback when OffscreenCanvas is unavailable, or in dev / simple
 * setups where adding a worker is premature.
 */

import { readImageInput } from '../image/input'
import type { ImagePixels } from '../domain/image'
import type { PageAsset } from '../domain/source'
import type { PreparedPageHandle, PageProjection } from '../domain/prepared'
import type { TextRegion } from '../domain/regions'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { PreparationStrategy } from '../domain/run'
import type { TextPlacement } from '../domain/planning'
import type { EncodedOcrImage } from '../recognizers/text'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import { estimateSafeMargins } from '../render/backgroundFit'
import { textFitRect } from '../render/fitGeometry'
import { CONTINUOUS_STRIP_PREPARE_PROFILE } from '../domain/prepare'
import { detectTextPresence } from '../pipeline/detectTextPresence'
import { BrowserCanvasBackend } from '../stages/prepare/canvasBackend'
import { detectSeam, type SeamAnalysis } from '../stages/prepare/seam/detectSeam'
import type { PreparationSession, VisionRuntime } from './VisionRuntime'

interface PreparedPage {
  readonly handle: PreparedPageHandle
  readonly image: ImagePixels
}

interface BufferedPage {
  readonly sourceIndex: number
  readonly image: ImagePixels
  readonly incomingSeam?: SeamAnalysis
}

const MAX_OCR_WIDTH = 1280
const MAX_OCR_HEIGHT = 9000
const DEFAULT_SEAM_REPAIR_BAND_PX = 320
const CONTINUOUS_STRIP_OVERLAP_PAGES = 1

export class MainThreadVisionRuntime implements VisionRuntime {
  private readonly runs = new Map<string, Map<string, PreparedPage>>()
  private readonly deps: {
    detector?: TextRegionDetector
  }

  constructor(deps: {
    detector?: TextRegionDetector
  }) {
    this.deps = deps
  }

  async beginPreparation(
    runId: string,
    strategy: PreparationStrategy,
  ): Promise<PreparationSession> {
    if (strategy.type === 'continuous-strip') {
      return new ContinuousStripPreparationSession(runId, this.runs)
    }
    return new MainThreadPreparationSession(
      runId,
      this.runs,
      strategy.type === 'identity-with-seams' ? strategy.seamBandPx ?? DEFAULT_SEAM_REPAIR_BAND_PX : null,
    )
  }

  async readPixels(handle: PreparedPageHandle): Promise<ImagePixels> {
    const page = this.requirePage(handle.runId, handle.preparedPageId)
    return {
      width: page.image.width,
      height: page.image.height,
      data: page.image.data,
    }
  }

  async encodeForOcr(handle: PreparedPageHandle, signal?: AbortSignal): Promise<EncodedOcrImage> {
    const page = this.requirePage(handle.runId, handle.preparedPageId)
    return encodeImageForOcr(page.image, signal)
  }

  async estimateMargins(
    handle: PreparedPageHandle,
    placements: readonly TextPlacement[],
  ): Promise<readonly SafeMarginsDebug[]> {
    const page = this.requirePage(handle.runId, handle.preparedPageId)
    return estimatePlacementMargins(page.image, placements)
  }

  async createSeamRepair(
    top: PreparedPageHandle,
    bottom: PreparedPageHandle,
    bandPx: number,
  ): Promise<PreparedPageHandle | null> {
    if (top.runId !== bottom.runId) throw new Error('seam repair handles belong to different runs')
    const run = this.requireRun(top.runId)
    const topPage = this.requirePage(top.runId, top.preparedPageId)
    const bottomPage = this.requirePage(bottom.runId, bottom.preparedPageId)
    const topProjection = top.projections[top.projections.length - 1]
    const bottomProjection = bottom.projections[0]
    if (!topProjection || !bottomProjection) return null

    const px = Math.max(1, Math.min(Math.round(bandPx), topPage.image.height, bottomPage.image.height))
    const topBand = cropBand(topPage.image, 'bottom', px)
    const bottomBand = cropBand(bottomPage.image, 'top', px)
    const image = await new BrowserCanvasBackend().stitchVertical([topBand, bottomBand])
    if (!detectTextPresence(image).hasText) return null

    const preparedPageIndex = run.size
    const preparedPageId = `${top.runId}:seam${preparedPageIndex}`
    const handle: PreparedPageHandle = {
      kind: 'seam-repair',
      runId: top.runId,
      preparedPageId,
      preparedPageIndex,
      size: { width: image.width, height: image.height },
      projections: [
        {
          sourcePageIndex: topProjection.sourcePageIndex,
          sourceRect: {
            x: 0,
            y: topProjection.sourceRect.y + topProjection.sourceRect.height - topBand.height,
            width: topBand.width,
            height: topBand.height,
          },
          sourcePageSize: topProjection.sourcePageSize ?? { width: topPage.image.width, height: topPage.image.height },
          preparedRect: { x: 0, y: 0, width: topBand.width, height: topBand.height },
        },
        {
          sourcePageIndex: bottomProjection.sourcePageIndex,
          sourceRect: {
            x: 0,
            y: bottomProjection.sourceRect.y,
            width: bottomBand.width,
            height: bottomBand.height,
          },
          sourcePageSize: bottomProjection.sourcePageSize ?? { width: bottomPage.image.width, height: bottomPage.image.height },
          preparedRect: { x: 0, y: topBand.height, width: bottomBand.width, height: bottomBand.height },
        },
      ],
    }
    run.set(preparedPageId, { handle, image })
    return handle
  }

  async detectTextRegions(
    handle: PreparedPageHandle,
    signal?: AbortSignal,
  ): Promise<readonly TextRegion[]> {
    const detector = this.deps.detector
    if (!detector) return []
    const page = this.requirePage(handle.runId, handle.preparedPageId)
    return detector.detectTextRegions(page.image, { signal })
  }

  release(handle: PreparedPageHandle): void {
    this.runs.get(handle.runId)?.delete(handle.preparedPageId)
  }

  cancelRun(runId: string): void {
    this.runs.delete(runId)
  }

  dispose(): void {
    this.runs.clear()
  }

  private requireRun(runId: string): Map<string, PreparedPage> {
    const run = this.runs.get(runId)
    if (!run) throw new Error(`vision run not found: ${runId}`)
    return run
  }

  private requirePage(runId: string, preparedPageId: string) {
    const page = this.requireRun(runId).get(preparedPageId)
    if (!page) throw new Error(`prepared page not found: ${preparedPageId}`)
    return page
  }
}

class MainThreadPreparationSession implements PreparationSession {
  private nextIndex = 0
  private previous: BufferedPage | null = null
  private readonly backend = new BrowserCanvasBackend()
  private readonly runId: string
  private readonly runs: Map<string, Map<string, PreparedPage>>
  private readonly seamBandPx: number | null

  constructor(
    runId: string,
    runs: Map<string, Map<string, PreparedPage>>,
    seamBandPx: number | null = null,
  ) {
    this.runId = runId
    this.runs = runs
    this.seamBandPx = seamBandPx
    this.runs.set(runId, new Map())
  }

  async push(asset: PageAsset): Promise<readonly PreparedPageHandle[]> {
    const image = asset.pixels ?? await readBlobPixels(asset)
    const handles: PreparedPageHandle[] = [
      this.storePreparedImage(image, asset.projections ?? fullPageProjection(asset.index, image), 'source-page'),
    ]

    if (this.seamBandPx !== null) {
      const previous = this.previous
      if (previous) {
        const seam = await this.prepareSeamRepairPage(previous, { sourceIndex: asset.index, image })
        if (seam) handles.push(seam)
      }
      this.previous = { sourceIndex: asset.index, image }
    }

    return handles
  }

  async flush(): Promise<readonly PreparedPageHandle[]> {
    this.previous = null
    return []
  }

  dispose(): void {
    this.previous = null
  }

  private async prepareSeamRepairPage(top: BufferedPage, bottom: BufferedPage): Promise<PreparedPageHandle | null> {
    const bandPx = Math.max(1, Math.min(Math.round(this.seamBandPx ?? DEFAULT_SEAM_REPAIR_BAND_PX), top.image.height, bottom.image.height))
    const topBand = cropBand(top.image, 'bottom', bandPx)
    const bottomBand = cropBand(bottom.image, 'top', bandPx)
    const image = await this.backend.stitchVertical([topBand, bottomBand])
    if (!detectTextPresence(image).hasText) return null
    return this.storePreparedImage(image, [
      {
        sourcePageIndex: top.sourceIndex,
        sourceRect: { x: 0, y: top.image.height - topBand.height, width: topBand.width, height: topBand.height },
        sourcePageSize: { width: top.image.width, height: top.image.height },
        preparedRect: { x: 0, y: 0, width: topBand.width, height: topBand.height },
      },
      {
        sourcePageIndex: bottom.sourceIndex,
        sourceRect: { x: 0, y: 0, width: bottomBand.width, height: bottomBand.height },
        sourcePageSize: { width: bottom.image.width, height: bottom.image.height },
        preparedRect: { x: 0, y: topBand.height, width: bottomBand.width, height: bottomBand.height },
      },
    ], 'seam-repair')
  }

  private storePreparedImage(
    image: ImagePixels,
    projections: readonly PageProjection[],
    kind: PreparedPageHandle['kind'],
  ): PreparedPageHandle {
    const preparedPageIndex = this.nextIndex++
    const preparedPageId = `${this.runId}:p${preparedPageIndex}`
    const handle: PreparedPageHandle = {
      kind,
      runId: this.runId,
      preparedPageId,
      preparedPageIndex,
      size: { width: image.width, height: image.height },
      projections,
    }
    this.runs.get(this.runId)!.set(preparedPageId, { handle, image })
    return handle
  }
}

async function readBlobPixels(asset: PageAsset): Promise<ImagePixels> {
  if (!asset.blob) throw new Error(`page ${asset.index} has no blob or pixels`)
  return readImageInput(asset.blob)
}

class ContinuousStripPreparationSession implements PreparationSession {
  private nextIndex = 0
  private buffer: BufferedPage[] = []
  private readonly backend = new BrowserCanvasBackend()
  private readonly runId: string
  private readonly runs: Map<string, Map<string, PreparedPage>>

  constructor(
    runId: string,
    runs: Map<string, Map<string, PreparedPage>>,
  ) {
    this.runId = runId
    this.runs = runs
    this.runs.set(runId, new Map())
  }

  async push(asset: PageAsset, signal?: AbortSignal): Promise<readonly PreparedPageHandle[]> {
    throwIfAborted(signal)
    const image = asset.pixels ?? await readBlobPixels(asset)
    await yieldToBrowser()

    if (!this.buffer.length) {
      this.replaceBuffer([{ sourceIndex: asset.index, image }])
      return []
    }

    const previous = this.buffer[this.buffer.length - 1]!
    const seam = await detectSeam({
      topSourcePageIndex: previous.sourceIndex,
      bottomSourcePageIndex: asset.index,
      top: previous.image,
      bottom: image,
      profile: CONTINUOUS_STRIP_PREPARE_PROFILE,
      backend: this.backend,
      signal,
    })
    await yieldToBrowser()

    const action = actionWithMemoryCaps(this.buffer, image, seam.decision.action)
    if (action === 'merge') {
      this.buffer.push({ sourceIndex: asset.index, image, incomingSeam: seam })
      return []
    }

    const emitted = await this.emitPreparedPage(signal)
    this.replaceBuffer(bufferAfterContinuousCut(this.buffer, { sourceIndex: asset.index, image, incomingSeam: seam }))
    return [emitted]
  }

  async flush(signal?: AbortSignal): Promise<readonly PreparedPageHandle[]> {
    throwIfAborted(signal)
    if (!this.buffer.length) return []
    const emitted = await this.emitPreparedPage(signal)
    this.clearBuffer()
    return [emitted]
  }

  dispose(): void {
    this.clearBuffer()
  }

  private async emitPreparedPage(signal?: AbortSignal): Promise<PreparedPageHandle> {
    const pages = this.buffer
    if (!pages.length) throw new Error('cannot emit empty prepared page')
    const preparedPageIndex = this.nextIndex++
    const preparedPageId = `${this.runId}:p${preparedPageIndex}`
    const image = pages.length === 1
      ? pages[0]!.image
      : await this.backend.stitchVertical(pages.map(page => page.image), signal)
    await yieldToBrowser()
    const handle: PreparedPageHandle = {
      kind: 'source-page',
      runId: this.runId,
      preparedPageId,
      preparedPageIndex,
      size: { width: image.width, height: image.height },
      projections: buildProjections(pages),
    }
    this.runs.get(this.runId)!.set(preparedPageId, { handle, image })
    return handle
  }

  private replaceBuffer(next: BufferedPage[]): void {
    this.buffer = next
  }

  private clearBuffer(): void {
    this.buffer = []
  }
}

function actionWithMemoryCaps(
  buffer: readonly BufferedPage[],
  next: ImagePixels,
  action: 'merge' | 'cut' | 'uncertain',
): 'merge' | 'cut' {
  const memory = CONTINUOUS_STRIP_PREPARE_PROFILE.memory
  const height = buffer.reduce((sum, page) => sum + page.image.height, 0) + next.height
  if (buffer.length >= memory.maxMergePages || height > memory.maxPreparedHeightPx) return 'cut'
  if (action === 'merge') return 'merge'
  if (action === 'cut') return 'cut'
  return CONTINUOUS_STRIP_PREPARE_PROFILE.seam.decision.uncertainAction
}

function bufferAfterContinuousCut(
  previous: readonly BufferedPage[],
  next: BufferedPage,
): BufferedPage[] {
  const carry = previous.slice(Math.max(0, previous.length - CONTINUOUS_STRIP_OVERLAP_PAGES))
  const withOverlap = [...carry, next]
  return fitsContinuousMemory(withOverlap) ? withOverlap : [next]
}

function fitsContinuousMemory(pages: readonly BufferedPage[]): boolean {
  const memory = CONTINUOUS_STRIP_PREPARE_PROFILE.memory
  const height = pages.reduce((sum, page) => sum + page.image.height, 0)
  return pages.length <= memory.maxMergePages && height <= memory.maxPreparedHeightPx
}

function buildProjections(pages: readonly BufferedPage[]): PageProjection[] {
  const projections: PageProjection[] = []
  let y = 0
  for (const page of pages) {
    projections.push({
      sourcePageIndex: page.sourceIndex,
      sourceRect: { x: 0, y: 0, width: page.image.width, height: page.image.height },
      sourcePageSize: { width: page.image.width, height: page.image.height },
      preparedRect: { x: 0, y, width: page.image.width, height: page.image.height },
    })
    y += page.image.height
  }
  return projections
}

function fullPageProjection(sourcePageIndex: number, image: ImagePixels): readonly PageProjection[] {
  return [
    {
      sourcePageIndex,
      sourceRect: { x: 0, y: 0, width: image.width, height: image.height },
      sourcePageSize: { width: image.width, height: image.height },
      preparedRect: { x: 0, y: 0, width: image.width, height: image.height },
    },
  ]
}

function cropBand(image: ImagePixels, edge: 'top' | 'bottom', bandPx: number): ImagePixels {
  const height = Math.max(1, Math.min(image.height, bandPx))
  const sourceY = edge === 'top' ? 0 : image.height - height
  const data = new Uint8ClampedArray(image.width * height * 4)
  const rowBytes = image.width * 4
  for (let y = 0; y < height; y++) {
    const sourceStart = (sourceY + y) * rowBytes
    data.set(image.data.subarray(sourceStart, sourceStart + rowBytes), y * rowBytes)
  }
  return { width: image.width, height, data }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0))
}

async function encodeImageForOcr(image: ImagePixels, signal?: AbortSignal): Promise<EncodedOcrImage> {
  throwIfAborted(signal)
  const scale = Math.min(1, MAX_OCR_WIDTH / image.width, MAX_OCR_HEIGHT / image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  const source = createCanvas(image.width, image.height)
  context2d(source).putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
  const out = createCanvas(width, height)
  context2d(out).drawImage(source, 0, 0, width, height)
  const blob = 'convertToBlob' in out
    ? await (out as OffscreenCanvas).convertToBlob({ type: 'image/png' })
    : await new Promise<Blob>((resolve, reject) => {
        ;(out as HTMLCanvasElement).toBlob(value => value ? resolve(value) : reject(new Error('failed to encode OCR image')), 'image/png')
      })
  return {
    bytes: new Uint8Array(await blob.arrayBuffer()),
    width,
    height,
    originalWidth: image.width,
    originalHeight: image.height,
    hasText: detectTextPresence(image).hasText || undefined,
  }
}

function estimatePlacementMargins(image: ImagePixels, placements: readonly TextPlacement[]): readonly SafeMarginsDebug[] {
  const pageSize: readonly [number, number] = [image.width, image.height]
  return placements.map((placement, index) => {
    const baseRect = textFitRect(placement)
    const others = placements
      .filter((_, i) => i !== index)
      .flatMap(p => p.textBoxes.length ? p.textBoxes : [p.bbox])
    return estimateSafeMargins({ image, placement, baseRect, obstacles: others, pageSize })
  })
}

function createCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

function context2d(canvas: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  return ctx
}
