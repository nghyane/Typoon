import type { PreparedChapter } from '../../domain/preparedChapter';
import type { PrepareRequest } from '../../domain/prepare';
import { type CanvasBackend } from './canvasBackend';
export declare function prepareChapter(request: PrepareRequest, backend?: CanvasBackend): Promise<PreparedChapter>;
