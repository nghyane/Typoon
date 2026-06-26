import type * as ort from 'onnxruntime-web/wasm'
import type { OrtModule } from './OrtBackend'

export type OrtProvider = 'webgpu' | 'wasm'

export interface OrtSessionHandle {
  readonly ort: OrtModule
  readonly session: ort.InferenceSession
  readonly provider: OrtProvider
  readonly descriptorId: string
}

export class OrtSessionPool {
  private readonly sessions = new Map<string, OrtSessionHandle>()

  constructor(private readonly ort: OrtModule) {}

  async session(descriptorId: string, bytes: ArrayBuffer, providers: readonly string[]): Promise<OrtSessionHandle> {
    const cached = this.sessions.get(descriptorId)
    if (cached) return cached

    const failures: string[] = []
    for (const provider of providers) {
      if (provider === 'webgpu' && !('gpu' in navigator)) {
        failures.push('webgpu: navigator.gpu unavailable')
        continue
      }

      try {
        const session = await this.ort.InferenceSession.create(bytes, { executionProviders: [provider] })
        const handle: OrtSessionHandle = { ort: this.ort, session, provider: provider as OrtProvider, descriptorId }
        this.sessions.set(descriptorId, handle)
        return handle
      } catch (error) {
        failures.push(`${provider}: ${error instanceof Error ? error.message : String(error)}`)
      }
    }

    throw new Error(`ORT session creation failed (${failures.join('; ')})`)
  }
}
