import type { PreparedChapter } from '../../domain/preparedChapter'
import type { PrepareRequest } from '../../domain/prepare'
import { BrowserCanvasBackend, type CanvasBackend } from './canvasBackend'
import { prepareIdentity } from './identity'
import { prepareContinuousStrip } from './continuousStrip'

export async function prepareChapter(
  request: PrepareRequest,
  backend: CanvasBackend = new BrowserCanvasBackend(),
): Promise<PreparedChapter> {
  if (request.strategy.type === 'identity') return prepareIdentity(request, backend)
  return prepareContinuousStrip(request, backend)
}
