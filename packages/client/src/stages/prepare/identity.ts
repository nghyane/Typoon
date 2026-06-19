import type { PrepareRequest } from '../../domain/prepare'
import type { PreparedChapter, PreparedPage, SourcePageMapping } from '../../domain/preparedChapter'
import type { CanvasBackend } from './canvasBackend'
import { MemoryPreparedPageAsset } from './asset'

export async function prepareIdentity(request: PrepareRequest, backend: CanvasBackend): Promise<PreparedChapter> {
  const pages: PreparedPage[] = []
  const mappings: SourcePageMapping[] = []

  for (let index = 0; index < request.source.pageCount; index++) {
    throwIfAborted(request.signal)
    const raw = await request.source.readPage(index, request.signal)
    if (!raw.blob) throw new Error(`source page ${raw.index} has no blob`)
    const image = await backend.decode(raw.blob, request.signal)
    const preparedIndex = pages.length

    pages.push({
      id: `${request.runId}:p${preparedIndex}`,
      index: preparedIndex,
      size: { width: image.width, height: image.height },
      asset: new MemoryPreparedPageAsset(image, backend),
      projections: [
        {
          sourcePageIndex: raw.index,
          sourceRect: { x: 0, y: 0, width: image.width, height: image.height },
          sourcePageSize: { width: image.width, height: image.height },
          preparedRect: { x: 0, y: 0, width: image.width, height: image.height },
        },
      ],
    })
    mappings.push({ sourcePageIndex: raw.index, preparedPageIndex: preparedIndex })
  }

  return { runId: request.runId, pages, sourcePageToPreparedPage: mappings }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
