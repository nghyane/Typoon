import type { OrtModule } from './OrtBackend'

export interface OrtRuntimeOptions {
  readonly wasmPaths?: string | { wasm?: string | URL; mjs?: string | URL }
  readonly wasmNumThreads?: number
  readonly logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal'
}

export class OrtRuntime {
  private configured = false

  constructor(private readonly ort: OrtModule) {}

  configure(options: OrtRuntimeOptions): void {
    if (this.configured) return

    if (typeof options.logLevel === 'string') this.ort.env.logLevel = options.logLevel
    if (options.wasmPaths) this.ort.env.wasm.wasmPaths = options.wasmPaths
    if (typeof options.wasmNumThreads === 'number') this.ort.env.wasm.numThreads = options.wasmNumThreads

    this.configured = true
  }
}
