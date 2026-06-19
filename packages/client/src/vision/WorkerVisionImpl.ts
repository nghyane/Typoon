/**
 * Worker-side implementation of VisionRuntime.
 *
 * All methods are called via Comlink proxy — no postMessage,
 * no requestId correlation, no switch/case dispatch.
 */
import * as Comlink from 'comlink'
import type { ImagePixels } from '../domain/image'
import type { PreparedPageHandle, PageProjection } from '../domain/prepared'
import type { PageAsset } from '../domain/source'
import type { TextRegion } from '../domain/regions'
import type { PreparationStrategy } from '../domain/run'
import type { TextPlacement } from '../domain/planning'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import type { EncodedOcrImage } from '../recognizers/text'
import { estimateSafeMargins } from '../render/backgroundFit'
import { textFitRect } from '../render/fitGeometry'
import { detectTextPresence } from '../pipeline/detectTextPresence'
import { CONTINUOUS_STRIP_PREPARE_PROFILE } from '../domain/prepare'
import { BrowserCanvasBackend } from '../stages/prepare/canvasBackend'
import { detectSeam, type SeamAnalysis } from '../stages/prepare/seam/detectSeam'
import type { PreparationSession } from './VisionRuntime'

const MAX_OCR_WIDTH = 1280
const MAX_OCR_HEIGHT = 9000
const DEFAULT_SEAM_REPAIR_BAND_PX = 320
const CONTINUOUS_STRIP_OVERLAP_PAGES = 1

type PreparedPage = {
  readonly handle: PreparedPageHandle
  readonly image: ImagePixels
}

type BufferedPage = {
  readonly sourceIndex: number
  readonly image: ImagePixels
  readonly incomingSeam?: SeamAnalysis
}

type RunState = {
  readonly strategy: PreparationStrategy
  nextPreparedIndex: number
  readonly pages: Map<string, PreparedPage>
  buffer: BufferedPage[]
}

export class WorkerVisionImpl {
  private readonly runs = new Map<string, RunState>()
  private readonly backend = new BrowserCanvasBackend()

  async beginPreparation(
    runId: string,
    strategy: PreparationStrategy,
  ): Promise<void> {
    console.debug('[vision-worker] beginPreparation', runId, JSON.stringify(strategy))
    this.runs.set(runId, {
      strategy,
      nextPreparedIndex: 0,
      pages: new Map(),
      buffer: [],
    })
  }

  async pushPreparation(
    runId: string,
    asset: PageAsset,
  ): Promise<readonly PreparedPageHandle[]> {
    return new WorkerPreparationSession(runId, this.runs, this.backend).push(asset)
  }

  async flushPreparation(
    runId: string,
  ): Promise<readonly PreparedPageHandle[]> {
    return new WorkerPreparationSession(runId, this.runs, this.backend).flush()
  }

  disposePreparation(runId: string): void {
    this.runs.delete(runId)
  }

  async readPixels(
    runId: string,
    preparedPageId: string,
  ): Promise<ImagePixels> {
    const page = requirePreparedPage(this.runs, runId, preparedPageId)
    const data = new Uint8ClampedArray(page.image.data)
    const result = { width: page.image.width, height: page.image.height, data }
    return Comlink.transfer(result, [data.buffer])
  }

  async encodeForOcr(
    runId: string,
    preparedPageId: string,
  ): Promise<EncodedOcrImage> {
    const page = requirePreparedPage(this.runs, runId, preparedPageId)
    const image = await encodeImageForOcr(page.image)
    const presence = detectTextPresence(page.image)
    console.debug(
      `[text-detect] page ${preparedPageId}:`,
      `hasText=${presence.hasText}`,
      `textBlocks=${Math.round(presence.textBlockFraction * 100)}%`,
      `size=${page.image.width}x${page.image.height}`,
    )
    const result = { ...image, hasText: presence.hasText || undefined }
    return Comlink.transfer(result, [image.bytes.buffer])
  }

  async estimateMargins(
    runId: string,
    preparedPageId: string,
    placements: readonly TextPlacement[],
  ): Promise<readonly SafeMarginsDebug[]> {
    const page = requirePreparedPage(this.runs, runId, preparedPageId)
    return estimatePlacementMargins(page.image, placements)
  }

  async createSeamRepair(
    runId: string,
    topPreparedPageId: string,
    bottomPreparedPageId: string,
    bandPx: number,
  ): Promise<PreparedPageHandle | null> {
    const run = requireRun(this.runs, runId)
    const top = requirePreparedPage(this.runs, runId, topPreparedPageId)
    const bottom = requirePreparedPage(this.runs, runId, bottomPreparedPageId)
    const topProjection = top.handle.projections[top.handle.projections.length - 1]
    const bottomProjection = bottom.handle.projections[0]
    if (!topProjection || !bottomProjection) return null

    const px = Math.max(1, Math.min(Math.round(bandPx), top.image.height, bottom.image.height))
    const topBand = cropBand(top.image, 'bottom', px)
    const bottomBand = cropBand(bottom.image, 'top', px)
    const image = await this.backend.stitchVertical([topBand, bottomBand])
    if (!detectTextPresence(image).hasText) return null

    return storePreparedImage(run, runId, image, [
      {
        sourcePageIndex: topProjection.sourcePageIndex,
        sourceRect: {
          x: 0,
          y: topProjection.sourceRect.y + topProjection.sourceRect.height - topBand.height,
          width: topBand.width,
          height: topBand.height,
        },
        sourcePageSize: topProjection.sourcePageSize ?? { width: top.image.width, height: top.image.height },
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
        sourcePageSize: bottomProjection.sourcePageSize ?? { width: bottom.image.width, height: bottom.image.height },
        preparedRect: { x: 0, y: topBand.height, width: bottomBand.width, height: bottomBand.height },
      },
    ], 'seam-repair')
  }

  async detectTextRegions(
    _runId: string,
    _preparedPageId: string,
  ): Promise<readonly TextRegion[]> {
    // ONNX detection runs on the main thread (WebGPU/WebGL).
    // The ComlinkVisionRuntime reads pixels and runs detection there.
    return []
  }

  release(runId: string, preparedPageId: string): void {
    this.runs.get(runId)?.pages.delete(preparedPageId)
  }

  cancelRun(runId: string): void {
    this.runs.delete(runId)
  }

  dispose(): void {
    this.runs.clear()
    self.close()
  }
}

// ── PreparationSession impl ──────────────────────────────────────────────

class WorkerPreparationSession implements PreparationSession {
  private readonly runId: string
  private readonly runs: Map<string, RunState>
  private readonly backend: BrowserCanvasBackend

  constructor(
    runId: string,
    runs: Map<string, RunState>,
    backend: BrowserCanvasBackend,
  ) {
    this.runId = runId
    this.runs = runs
    this.backend = backend
  }

  async push(asset: PageAsset): Promise<readonly PreparedPageHandle[]> {
    const run = requireRun(this.runs, this.runId)
    const strategyType = (run.strategy as { type?: string }).type
    console.debug('[vision-worker] push', this.runId, 'strategy=', strategyType, 'asset.index=', asset.index, 'hasBlob=', !!asset.blob)
    if (strategyType === 'identity') {
      return prepareIdentityPage(this.runs, run, this.runId, asset)
    }
    if (strategyType === 'identity-with-seams') {
      return prepareIdentityWithSeamsPage(this.runs, run, this.runId, asset, this.backend)
    }
    if (strategyType === 'continuous-strip') {
      return prepareContinuousStripPage(this.runs, run, this.runId, asset, this.backend)
    }
    throw new Error(`unsupported preparation strategy: ${strategyType}`)
  }

  async flush(): Promise<readonly PreparedPageHandle[]> {
    const run = requireRun(this.runs, this.runId)
    const strategyType = (run.strategy as { type?: string }).type
    if (strategyType === 'identity-with-seams') {
      run.buffer = []
      return []
    }
    if (strategyType !== 'continuous-strip') return []
    if (!run.buffer.length) return []
    const emitted = await emitContinuousPreparedPage(this.runs, run, this.runId, this.backend)
    run.buffer = []
    return [emitted]
  }

  dispose(): void {
    this.runs.delete(this.runId)
  }
}

// ── Preparation helpers ──────────────────────────────────────────────────

async function prepareIdentityPage(
  runs: Map<string, RunState>,
  run: RunState,
  runId: string,
  asset: { readonly index: number; readonly blob?: Blob },
): Promise<readonly PreparedPageHandle[]> {
  if (!asset.blob) throw new Error(`page ${asset.index} has no blob`)
  const image = await decodeBlob(asset.blob)
  assertRunActive(runs, runId, run)
  return [storePreparedImage(run, runId, image, fullPageProjection(asset.index, image), 'source-page')]
}

async function prepareIdentityWithSeamsPage(
  runs: Map<string, RunState>,
  run: RunState,
  runId: string,
  asset: { readonly index: number; readonly blob?: Blob },
  backend: BrowserCanvasBackend,
): Promise<readonly PreparedPageHandle[]> {
  if (!asset.blob) throw new Error(`page ${asset.index} has no blob`)
  const image = await decodeBlob(asset.blob)
  assertRunActive(runs, runId, run)

  const handles: PreparedPageHandle[] = [
    storePreparedImage(run, runId, image, fullPageProjection(asset.index, image), 'source-page'),
  ]

  const previous = run.buffer[run.buffer.length - 1]
  if (previous) {
    const seam = await prepareSeamRepairPage(run, runId, previous, { sourceIndex: asset.index, image }, backend)
    assertRunActive(runs, runId, run)
    if (seam) handles.push(seam)
  }

  run.buffer = [{ sourceIndex: asset.index, image }]
  return handles
}

async function prepareContinuousStripPage(
  runs: Map<string, RunState>,
  run: RunState,
  runId: string,
  asset: { readonly index: number; readonly blob?: Blob },
  backend: BrowserCanvasBackend,
): Promise<readonly PreparedPageHandle[]> {
  if (!asset.blob) throw new Error(`page ${asset.index} has no blob`)
  const image = await decodeBlob(asset.blob)
  assertRunActive(runs, runId, run)

  if (!run.buffer.length) {
    run.buffer = [{ sourceIndex: asset.index, image }]
    return []
  }

  const previous = run.buffer[run.buffer.length - 1]!
  const seam = await detectSeam({
    topSourcePageIndex: previous.sourceIndex,
    bottomSourcePageIndex: asset.index,
    top: previous.image,
    bottom: image,
    profile: CONTINUOUS_STRIP_PREPARE_PROFILE,
    backend,
  })
  assertRunActive(runs, runId, run)

  const action = actionWithMemoryCaps(run.buffer, image, seam.decision.action)
  if (action === 'merge') {
    run.buffer.push({ sourceIndex: asset.index, image, incomingSeam: seam })
    return []
  }

  const emitted = await emitContinuousPreparedPage(runs, run, runId, backend)
  run.buffer = bufferAfterContinuousCut(run.buffer, { sourceIndex: asset.index, image, incomingSeam: seam })
  return [emitted]
}

async function emitContinuousPreparedPage(
  runs: Map<string, RunState>,
  run: RunState,
  runId: string,
  backend: BrowserCanvasBackend,
): Promise<PreparedPageHandle> {
  const pages = run.buffer
  if (!pages.length) throw new Error('cannot emit empty prepared page')
  const image = pages.length === 1
    ? pages[0]!.image
    : await backend.stitchVertical(pages.map(page => page.image))
  assertRunActive(runs, runId, run)
  const preparedPageIndex = run.nextPreparedIndex++
  const preparedPageId = `${runId}:p${preparedPageIndex}`
  const handle: PreparedPageHandle = {
    kind: 'source-page',
    runId,
    preparedPageId,
    preparedPageIndex,
    size: { width: image.width, height: image.height },
    projections: buildProjections(pages),
  }
  run.pages.set(preparedPageId, { handle, image })
  return handle
}

async function prepareSeamRepairPage(
  run: RunState,
  runId: string,
  top: BufferedPage,
  bottom: BufferedPage,
  backend: BrowserCanvasBackend,
): Promise<PreparedPageHandle | null> {
  const bandPx = seamRepairBandPx(top.image, bottom.image, run.strategy)
  const topBand = cropBand(top.image, 'bottom', bandPx)
  const bottomBand = cropBand(bottom.image, 'top', bandPx)
  const image = await backend.stitchVertical([topBand, bottomBand])
  if (!detectTextPresence(image).hasText) return null

  return storePreparedImage(run, runId, image, [
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

function storePreparedImage(
  run: RunState,
  runId: string,
  image: ImagePixels,
  projections: readonly PageProjection[],
  kind: PreparedPageHandle['kind'],
): PreparedPageHandle {
  const preparedPageIndex = run.nextPreparedIndex++
  const preparedPageId = `${runId}:p${preparedPageIndex}`
  const handle: PreparedPageHandle = {
    kind,
    runId,
    preparedPageId,
    preparedPageIndex,
    size: { width: image.width, height: image.height },
    projections,
  }
  run.pages.set(preparedPageId, { handle, image })
  return handle
}

function fullPageProjection(sourcePageIndex: number, image: ImagePixels): readonly PageProjection[] {
  return [
    {
      sourcePageIndex,
      preparedRect: { x: 0, y: 0, width: image.width, height: image.height },
      sourceRect: { x: 0, y: 0, width: image.width, height: image.height },
      sourcePageSize: { width: image.width, height: image.height },
    },
  ]
}

function seamRepairBandPx(top: ImagePixels, bottom: ImagePixels, strategy: PreparationStrategy): number {
  const requested = strategy.type === 'identity-with-seams' ? strategy.seamBandPx ?? DEFAULT_SEAM_REPAIR_BAND_PX : DEFAULT_SEAM_REPAIR_BAND_PX
  return Math.max(1, Math.min(Math.round(requested), top.height, bottom.height))
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

// ── Helpers ──────────────────────────────────────────────────────────────

function requireRun(runs: Map<string, RunState>, runId: string): RunState {
  const run = runs.get(runId)
  if (!run) throw new Error(`vision run not found: ${runId}`)
  return run
}

function assertRunActive(runs: Map<string, RunState>, runId: string, run: RunState): void {
  if (runs.get(runId) === run) return
  throw new Error(`vision run cancelled: ${runId}`)
}

function requirePreparedPage(
  runs: Map<string, RunState>,
  runId: string,
  preparedPageId: string,
): PreparedPage {
  const page = requireRun(runs, runId).pages.get(preparedPageId)
  if (!page) throw new Error(`prepared page not found: ${preparedPageId}`)
  return page
}

async function decodeBlob(blob: Blob): Promise<ImagePixels> {
  const bitmap = await createImageBitmap(blob)
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('2d canvas unavailable in vision worker')
    ctx.drawImage(bitmap, 0, 0)
    const image = ctx.getImageData(0, 0, bitmap.width, bitmap.height)
    return { width: image.width, height: image.height, data: image.data }
  } finally {
    bitmap.close()
  }
}

async function encodeImageForOcr(image: ImagePixels): Promise<EncodedOcrImage> {
  const scale = Math.min(1, MAX_OCR_WIDTH / image.width, MAX_OCR_HEIGHT / image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  const source = new OffscreenCanvas(image.width, image.height)
  const sourceCtx = source.getContext('2d')
  if (!sourceCtx) throw new Error('2d canvas unavailable in vision worker')
  sourceCtx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
  const out = new OffscreenCanvas(width, height)
  const outCtx = out.getContext('2d')
  if (!outCtx) throw new Error('2d canvas unavailable in vision worker')
  outCtx.drawImage(source, 0, 0, width, height)
  const blob = await out.convertToBlob({ type: 'image/png' })
  return {
    bytes: new Uint8Array(await blob.arrayBuffer()),
    width,
    height,
    originalWidth: image.width,
    originalHeight: image.height,
  }
}

function estimatePlacementMargins(
  image: ImagePixels,
  placements: readonly TextPlacement[],
): readonly SafeMarginsDebug[] {
  const pageSize: readonly [number, number] = [image.width, image.height]
  return placements.map((placement, index) => {
    const baseRect = textFitRect(placement)
    const others = placements
      .filter((_, i) => i !== index)
      .flatMap(p => p.textBoxes.length ? p.textBoxes : [p.bbox])
    return estimateSafeMargins({ image, placement, baseRect, obstacles: others, pageSize })
  })
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
