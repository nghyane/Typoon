// detectors/manga/ortWorkerProtocol.ts — Comlink contract between the main
// thread and the ORT inference worker. Only plain/transferable data crosses.

export interface OrtWorkerInit {
  readonly bytes: ArrayBuffer
  readonly provider: 'webgpu' | 'wasm'
  readonly wasmPaths: { readonly wasm: string; readonly mjs: string }
  readonly numThreads: number
}

export interface OrtRawTensor {
  readonly data: ArrayBufferView
  readonly dims: readonly number[]
  readonly type: string
}

export interface OrtRunResult {
  readonly outputs: Record<string, OrtRawTensor>
  readonly outputNames: readonly string[]
}

export interface OrtRunArgs {
  readonly images: Float32Array
  readonly width: number
  readonly height: number
}

export interface OrtWorkerApi {
  init(args: OrtWorkerInit): Promise<readonly string[]>
  run(args: OrtRunArgs): Promise<OrtRunResult>
}
