import * as ort from 'onnxruntime-web/webgpu';
import type { ImagePixels } from '../../domain/image';
export declare function createFeeds(image: ImagePixels): Record<string, ort.Tensor>;
