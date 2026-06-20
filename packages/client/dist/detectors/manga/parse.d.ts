import type * as ort from 'onnxruntime-web/wasm';
import type { TextRegion } from '../../domain/regions';
export declare function parseDetections(output: Record<string, ort.Tensor>, outputNames: readonly string[], pageW: number, pageH: number, confidenceThreshold: number): TextRegion[];
