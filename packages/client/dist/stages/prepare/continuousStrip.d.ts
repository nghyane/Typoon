import type { PrepareRequest } from '../../domain/prepare';
import type { PreparedChapter } from '../../domain/preparedChapter';
import type { CanvasBackend } from './canvasBackend';
export declare function prepareContinuousStrip(request: PrepareRequest, backend: CanvasBackend): Promise<PreparedChapter>;
