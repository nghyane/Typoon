import type { ImagePixels } from '../../domain/image'
import type { PrepareRequest } from '../../domain/prepare'
import type { PageProjection, PreparedChapter, PreparedPage, SourcePageMapping } from '../../domain/preparedChapter'
import type { CanvasBackend } from './canvasBackend'
import { MemoryPreparedPageAsset } from './asset'
import { detectSeam, type SeamAnalysis } from './seam/detectSeam'

interface BufferedPage {
  readonly sourceIndex: number
  readonly image: ImagePixels
  readonly incomingSeam?: SeamAnalysis
}

const CONTINUOUS_STRIP_OVERLAP_PAGES = 1

export async function prepareContinuousStrip(request: PrepareRequest, backend: CanvasBackend): Promise<PreparedChapter> {
  const pages: PreparedPage[] = []
  const mappings: SourcePageMapping[] = []
  let buffer: BufferedPage[] = []

  for (let index = 0; index < request.source.pageCount; index++) {
    throwIfAborted(request.signal)
    const raw = await request.source.readPage(index, request.signal)
    if (!raw.blob) throw new Error(`source page ${raw.index} has no blob`)
    await yieldToBrowser()
    const image = await backend.decode(raw.blob, request.signal)
    await yieldToBrowser()

    if (!buffer.length) {
      buffer = [{ sourceIndex: raw.index, image }]
      continue
    }

    const previous = buffer[buffer.length - 1]!
    const seam = await detectSeam({
      topSourcePageIndex: previous.sourceIndex,
      bottomSourcePageIndex: raw.index,
      top: previous.image,
      bottom: image,
      profile: request.profile,
      backend,
      signal: request.signal,
    })
    await yieldToBrowser()
    await request.artifacts?.writeJson(`01_prepare/seams/${previous.sourceIndex}_${raw.index}.json`, seam)

    const action = actionWithMemoryCaps({
      action: seam.decision.action,
      buffer,
      next: image,
      request,
    })

    if (action === 'merge') {
      buffer.push({ sourceIndex: raw.index, image, incomingSeam: seam })
      continue
    }

    const prepared = await emitPreparedPage(request, backend, pages.length, buffer)
    pages.push(prepared)
    for (const projection of prepared.projections) {
      mappings.push({ sourcePageIndex: projection.sourcePageIndex, preparedPageIndex: prepared.index })
    }
    buffer = bufferAfterContinuousCut(buffer, { sourceIndex: raw.index, image, incomingSeam: seam }, request.profile)
  }

  if (buffer.length) {
    const prepared = await emitPreparedPage(request, backend, pages.length, buffer)
    pages.push(prepared)
    for (const projection of prepared.projections) {
      mappings.push({ sourcePageIndex: projection.sourcePageIndex, preparedPageIndex: prepared.index })
    }
  }

  return { runId: request.runId, pages, sourcePageToPreparedPage: mappings }
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0))
}

async function emitPreparedPage(
  request: PrepareRequest,
  backend: CanvasBackend,
  preparedIndex: number,
  pages: readonly BufferedPage[],
): Promise<PreparedPage> {
  const image = pages.length === 1
    ? pages[0]!.image
    : await backend.stitchVertical(pages.map(page => page.image), request.signal)
  const prepared: PreparedPage = {
    id: `${request.runId}:p${preparedIndex}`,
    index: preparedIndex,
    size: { width: image.width, height: image.height },
    asset: new MemoryPreparedPageAsset(image, backend),
    projections: buildProjections(pages),
  }
  await request.artifacts?.writeJson(`01_prepare/pages/${preparedIndex}.json`, {
    id: prepared.id,
    index: prepared.index,
    size: prepared.size,
    projections: prepared.projections,
    seams: pages.map(page => page.incomingSeam).filter(Boolean),
  })
  await request.artifacts?.writeImage(`01_prepare/pages/${preparedIndex}.png`, image)
  return prepared
}

function actionWithMemoryCaps(args: {
  readonly action: 'merge' | 'cut' | 'uncertain'
  readonly buffer: readonly BufferedPage[]
  readonly next: ImagePixels
  readonly request: PrepareRequest
}): 'merge' | 'cut' {
  const height = bufferHeight(args.buffer) + args.next.height
  const memory = args.request.profile.memory
  if (args.buffer.length >= memory.maxMergePages || height > memory.maxPreparedHeightPx) return 'cut'
  if (args.action === 'merge') return 'merge'
  if (args.action === 'cut') return 'cut'
  return args.request.profile.seam.decision.uncertainAction
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

function bufferHeight(buffer: readonly BufferedPage[]): number {
  return buffer.reduce((sum, page) => sum + page.image.height, 0)
}

function bufferAfterContinuousCut(
  previous: readonly BufferedPage[],
  next: BufferedPage,
  profile: PrepareRequest['profile'],
): BufferedPage[] {
  const carry = previous.slice(Math.max(0, previous.length - CONTINUOUS_STRIP_OVERLAP_PAGES))
  const withOverlap = [...carry, next]
  return fitsContinuousMemory(withOverlap, profile) ? withOverlap : [next]
}

function fitsContinuousMemory(pages: readonly BufferedPage[], profile: PrepareRequest['profile']): boolean {
  const memory = profile.memory
  return pages.length <= memory.maxMergePages && bufferHeight(pages) <= memory.maxPreparedHeightPx
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
