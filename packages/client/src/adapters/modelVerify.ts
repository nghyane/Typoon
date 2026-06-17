import type { ModelDescriptor } from './modelTypes'

export async function verifyModel(bytes: ArrayBuffer, descriptor: ModelDescriptor): Promise<void> {
  if (descriptor.sizeBytes && bytes.byteLength !== descriptor.sizeBytes) {
    throw new Error(`model size mismatch: expected ${descriptor.sizeBytes}, got ${bytes.byteLength}`)
  }
  if (!descriptor.sha256) return
  const hash = await sha256(bytes)
  if (hash !== descriptor.sha256.toLowerCase()) {
    throw new Error(`model sha256 mismatch: expected ${descriptor.sha256}, got ${hash}`)
  }
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}
