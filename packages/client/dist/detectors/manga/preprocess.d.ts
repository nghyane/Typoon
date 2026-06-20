import type * as ort from 'onnxruntime-web/wasm';
import type { ImagePixels } from '../../domain/image';
import type { OrtModule } from '../../models/OrtBackend';
export declare function createFeeds(ortModule: OrtModule, image: ImagePixels): Record<string, ort.Tensor>;
