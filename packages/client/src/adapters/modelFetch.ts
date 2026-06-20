import type { CapabilityProgress } from '../domain/capability'
import type { ModelDescriptor } from './modelTypes'
import { verifyModel } from './modelVerify'

const RANGE_DOWNLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
const RANGE_CHUNK_BYTES = 8 * 1024 * 1024
const PROGRESS_INTERVAL_MS = 100

export async function loadModelBytes(
  descriptor: ModelDescriptor,
  onProgress: (progress: CapabilityProgress) => void,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  throwIfAborted(signal)

  onProgress({ receivedBytes: 0, totalBytes: descriptor.sizeBytes, ratio: 0 })
  if (descriptor.sizeBytes >= RANGE_DOWNLOAD_THRESHOLD_BYTES) {
    return loadModelBytesInRanges(descriptor, onProgress, signal)
  }

  const response = await fetch(descriptor.url, { mode: 'cors', signal })
  if (!response.ok) throw new Error(`model fetch failed: ${response.status}`)
  return readAndVerifyModelResponse(descriptor, response, onProgress, signal)
}

async function loadModelBytesInRanges(
  descriptor: ModelDescriptor,
  onProgress: (progress: CapabilityProgress) => void,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  const out = new Uint8Array(descriptor.sizeBytes)
  const progress = createProgressReporter(descriptor.sizeBytes, onProgress)

  for (let start = 0; start < descriptor.sizeBytes; start += RANGE_CHUNK_BYTES) {
    throwIfAborted(signal)
    const end = Math.min(start + RANGE_CHUNK_BYTES, descriptor.sizeBytes) - 1
    const response = await fetch(descriptor.url, {
      mode: 'cors',
      signal,
      headers: { Range: `bytes=${start}-${end}` },
    })
    if (response.status === 200 && start === 0) {
      return readAndVerifyModelResponse(descriptor, response, onProgress, signal)
    }
    if (response.status !== 206) throw new Error(`model range fetch failed: ${response.status}`)
    assertContentRange(response, start, end, descriptor.sizeBytes)

    const expectedBytes = end - start + 1
    const writtenBytes = await readResponseInto(response, out, start, signal, bytes => progress.add(bytes))
    if (writtenBytes !== expectedBytes) throw new Error(`model range fetch length mismatch: ${writtenBytes}/${expectedBytes}`)
  }

  progress.done()
  const bytes = out.buffer
  throwIfAborted(signal)
  await verifyModel(bytes, descriptor)
  return bytes
}

async function readAndVerifyModelResponse(
  descriptor: ModelDescriptor,
  response: Response,
  onProgress: (progress: CapabilityProgress) => void,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  const bytes = await readResponseBytes(response, signal, onProgress)
  throwIfAborted(signal)
  await verifyModel(bytes, descriptor)
  return bytes
}

async function readResponseInto(
  response: Response,
  out: Uint8Array,
  offset: number,
  signal: AbortSignal | undefined,
  onChunk: (bytes: number) => void,
): Promise<number> {
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer())
    throwIfAborted(signal)
    if (offset + bytes.byteLength > out.byteLength) throw new Error('model fetch exceeded expected size')
    out.set(bytes, offset)
    onChunk(bytes.byteLength)
    return bytes.byteLength
  }

  let writtenBytes = 0
  const reader = response.body.getReader()
  while (true) {
    throwIfAborted(signal)
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    if (offset + writtenBytes + value.byteLength > out.byteLength) throw new Error('model fetch exceeded expected size')
    out.set(value, offset + writtenBytes)
    writtenBytes += value.byteLength
    onChunk(value.byteLength)
  }
  return writtenBytes
}

async function readResponseBytes(
  response: Response,
  signal: AbortSignal | undefined,
  onProgress: (progress: CapabilityProgress) => void,
): Promise<ArrayBuffer> {
  const totalBytes = Number(response.headers.get('Content-Length') ?? '') || undefined
  if (!response.body) {
    const bytes = await response.arrayBuffer()
    throwIfAborted(signal)
    onProgress({ receivedBytes: bytes.byteLength, totalBytes, ratio: totalBytes ? bytes.byteLength / totalBytes : undefined })
    return bytes
  }

  if (totalBytes !== undefined) {
    const out = new Uint8Array(totalBytes)
    const progress = createProgressReporter(totalBytes, onProgress)
    const writtenBytes = await readResponseInto(response, out, 0, signal, bytes => progress.add(bytes))
    if (writtenBytes !== totalBytes) throw new Error(`model fetch length mismatch: ${writtenBytes}/${totalBytes}`)
    progress.done()
    return out.buffer
  }

  const chunks: Uint8Array[] = []
  let receivedBytes = 0
  const progress = createProgressReporter(undefined, onProgress)
  const reader = response.body.getReader()
  while (true) {
    throwIfAborted(signal)
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    receivedBytes += value.byteLength
    progress.add(value.byteLength)
  }
  const out = new Uint8Array(receivedBytes)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  progress.done()
  return out.buffer
}

function assertContentRange(response: Response, start: number, end: number, totalBytes: number): void {
  const contentRange = response.headers.get('Content-Range')
  if (!contentRange) return
  const expected = `bytes ${start}-${end}/${totalBytes}`
  if (contentRange !== expected) throw new Error(`model range mismatch: expected ${expected}, got ${contentRange}`)
}

function createProgressReporter(
  totalBytes: number | undefined,
  onProgress: (progress: CapabilityProgress) => void,
): { add(bytes: number): void; done(): void } {
  let receivedBytes = 0
  let lastEmit = 0

  function emit(force: boolean): void {
    const ratio = totalBytes ? receivedBytes / totalBytes : undefined
    const now = performance.now()
    if (!force && now - lastEmit < PROGRESS_INTERVAL_MS && ratio !== 1) return
    lastEmit = now
    onProgress({ receivedBytes, totalBytes, ratio })
  }

  return {
    add(bytes: number): void {
      receivedBytes += bytes
      emit(false)
    },
    done(): void {
      emit(true)
    },
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
