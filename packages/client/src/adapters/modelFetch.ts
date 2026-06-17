import type { CapabilityProgress } from '../domain/capability'
import type { ModelAssetCache, ModelDescriptor } from './modelTypes'
import { verifyModel } from './modelVerify'

export async function loadModelBytes(
  descriptor: ModelDescriptor,
  cache: ModelAssetCache,
  onProgress: (progress: CapabilityProgress) => void,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  throwIfAborted(signal)
  const cached = await readCachedModel(cache, descriptor.url)
  throwIfAborted(signal)
  if (cached) {
    try {
      await verifyModel(cached, descriptor)
      throwIfAborted(signal)
      return cached
    } catch (error) {
      console.warn('[typoon-client] cached model failed verification; fetching network copy', error)
    }
  }

  throwIfAborted(signal)
  onProgress({ receivedBytes: 0, totalBytes: descriptor.sizeBytes, ratio: 0 })
  const response = await fetch(descriptor.url, { mode: 'cors', signal })
  if (!response.ok) throw new Error(`model fetch failed: ${response.status}`)
  const bytes = await readResponseBytes(response, signal, onProgress)
  throwIfAborted(signal)
  await verifyModel(bytes, descriptor)
  throwIfAborted(signal)
  if (!signal) await writeCachedModel(cache, descriptor.url, bytes)
  return bytes
}

async function readCachedModel(cache: ModelAssetCache, key: string): Promise<ArrayBuffer | null> {
  try {
    return await cache.match(key)
  } catch (error) {
    console.warn('[typoon-client] model cache read failed; fetching network copy', error)
    return null
  }
}

async function writeCachedModel(cache: ModelAssetCache, key: string, bytes: ArrayBuffer): Promise<void> {
  try {
    await cache.put(key, bytes)
  } catch (error) {
    console.warn('[typoon-client] model cache write failed; continuing with memory copy', error)
  }
}

async function readResponseBytes(
  response: Response,
  signal: AbortSignal | undefined,
  onProgress: (progress: { receivedBytes: number; totalBytes?: number; ratio?: number }) => void,
): Promise<ArrayBuffer> {
  const totalBytes = Number(response.headers.get('Content-Length') ?? '') || undefined
  if (!response.body) {
    const bytes = await response.arrayBuffer()
    throwIfAborted(signal)
    onProgress({ receivedBytes: bytes.byteLength, totalBytes, ratio: totalBytes ? bytes.byteLength / totalBytes : undefined })
    return bytes
  }

  const chunks: Uint8Array[] = []
  let receivedBytes = 0
  let lastEmit = 0
  const reader = response.body.getReader()
  while (true) {
    throwIfAborted(signal)
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    receivedBytes += value.byteLength
    const now = performance.now()
    const ratio = totalBytes ? receivedBytes / totalBytes : undefined
    if (now - lastEmit >= 100 || (ratio !== undefined && ratio >= 1)) {
      lastEmit = now
      onProgress({ receivedBytes, totalBytes, ratio })
    }
  }
  const out = new Uint8Array(receivedBytes)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  onProgress({ receivedBytes, totalBytes, ratio: totalBytes ? receivedBytes / totalBytes : undefined })
  return out.buffer
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
