// detectors/manga/ortWorker.ts — runs Comic-DETR inference off the main thread.
//
// The heavy, blocking part is session.run(); keeping it here stops it from
// janking the UI while pages pipeline. Preprocessing (canvas resize) stays on
// the main thread, so only the planar tensor crosses in.

import * as Comlink from 'comlink'
import { COMIC_DETR_INPUT_SIZE } from './ortTypes'
import { AsyncLimiter } from '../../flow/AsyncLimiter'
import type { OrtRunArgs, OrtRunResult, OrtWorkerApi, OrtWorkerInit } from './ortWorkerProtocol'

// onnxruntime-web is loaded dynamically per chosen provider; typed loosely here
// because the worker only touches Tensor/InferenceSession surface.
type OrtLike = {
  env: { logLevel: string; wasm: { wasmPaths: unknown; numThreads: number } }
  Tensor: new (type: string, data: ArrayBufferView, dims: readonly number[]) => unknown
  InferenceSession: { create(bytes: ArrayBuffer, opts: unknown): Promise<OrtSessionLike> }
}
interface OrtSessionLike {
  readonly outputNames: readonly string[]
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: ArrayBufferView; dims: readonly number[]; type: string }>>
}

let mod: OrtLike | null = null
let session: OrtSessionLike | null = null
let outputNames: readonly string[] = []
// A single InferenceSession is not reentrant; comlink dispatches run() calls
// concurrently, so serialize them here.
const inferenceLock = new AsyncLimiter(1)

const api: OrtWorkerApi = {
  async init({ bytes, provider, wasmPaths, numThreads }: OrtWorkerInit): Promise<readonly string[]> {
    mod = (provider === 'webgpu'
      ? await import('onnxruntime-web/webgpu')
      : await import('onnxruntime-web/wasm')) as unknown as OrtLike
    mod.env.logLevel = 'fatal'
    mod.env.wasm.wasmPaths = wasmPaths
    mod.env.wasm.numThreads = numThreads
    if (provider === 'webgpu' && !('gpu' in (globalThis.navigator ?? {}))) {
      throw new Error('webgpu unavailable in worker')
    }
    session = await mod.InferenceSession.create(bytes, { executionProviders: [provider] })
    outputNames = session.outputNames
    return outputNames
  },

  run(args: OrtRunArgs): Promise<OrtRunResult> {
    return inferenceLock.run(async () => runInference(args))
  },
}

async function runInference({ images }: OrtRunArgs): Promise<OrtRunResult> {
    if (!session || !mod) throw new Error('ort worker not initialized')
    // Raw model takes only `images`; boxes are scaled by page size in
    // parseDetections (on the main thread), so width/height aren't fed here.
    const feeds = {
      images: new mod.Tensor('float32', images, [1, 3, COMIC_DETR_INPUT_SIZE, COMIC_DETR_INPUT_SIZE]),
    }
    const out = await session.run(feeds)
    const outputs: Record<string, { data: ArrayBufferView; dims: readonly number[]; type: string }> = {}
    const transfers: Transferable[] = []
    for (const name of outputNames) {
      const tensor = out[name]
      if (!tensor) continue
      outputs[name] = { data: tensor.data, dims: tensor.dims, type: tensor.type }
      if (tensor.data.buffer instanceof ArrayBuffer) transfers.push(tensor.data.buffer)
    }
    return Comlink.transfer({ outputs, outputNames }, transfers)
}

Comlink.expose(api)
