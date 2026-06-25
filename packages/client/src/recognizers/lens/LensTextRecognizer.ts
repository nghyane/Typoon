import type { BBox, OrientedBox, Point, Polygon } from '../../domain/geometry'
import type { ImagePixels } from '../../domain/image'
import type { TextBlock, TextLine, RecognizedTextPage, TextWord, TextDirection } from '../../domain/text'
import type { EncodedOcrImage, TextRecognizer, TextRecognitionOptions } from '../text'
import { AsyncLimiter } from '../../flow/AsyncLimiter'
import * as lens from 'chrome-lens-ocr/src/utils/proto_generated/lens_overlay_server_pb.cjs'

const DEFAULT_ENDPOINT = 'https://927251094806098001.discordsays.com/lens/v1/crupload'
const LENS_API_KEY = 'AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY'
const MAX_LENS_WIDTH = 1280
const MAX_LENS_HEIGHT = 9000
const DEFAULT_REQUEST_TIMEOUT_MS = 20_000
const MAX_REQUEST_ATTEMPTS = 3
const RETRY_BASE_DELAY_MS = 600
// Global cap on concurrent Lens round-trips. Page-pipelining fans out one
// full-page OCR plus several bubble-recovery crops per page across several
// pages at once; bound them at the shared resource so the proxy is not flooded
// (429), instead of per-call limiters that don't see each other.
const DEFAULT_MAX_CONCURRENCY = 8

export interface LensTextRecognizerOptions {
  readonly endpoint?: string
  readonly requestTimeoutMs?: number
  readonly region?: string
  readonly timeZone?: string
  readonly maxConcurrency?: number
}

export class LensTextRecognizer implements TextRecognizer {
  readonly name = 'lens-text-recognizer'
  private readonly options: LensTextRecognizerOptions
  private readonly endpoint: string
  private readonly requestTimeoutMs: number
  private readonly limiter: AsyncLimiter

  constructor(options: LensTextRecognizerOptions = {}) {
    this.options = options
    this.endpoint = options.endpoint ?? DEFAULT_ENDPOINT
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
    this.limiter = new AsyncLimiter(options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY)
  }

  async recognizeText(image: ImagePixels, options: TextRecognitionOptions): Promise<RecognizedTextPage> {
    return this.recognizeEncoded(await encodeImageForLens(image), options)
  }

  async recognizeEncoded(image: EncodedOcrImage, options: TextRecognitionOptions): Promise<RecognizedTextPage> {
    const start = performance.now()
    if (image.hasText === false) {
      return {
        pageIndex: options.pageIndex,
        pageSize: [image.originalWidth, image.originalHeight],
        detectedLanguage: null,
        blocks: [],
        timingMs: { recognize: Math.round(performance.now() - start) },
      }
    }

    const request = createLensRequest({
      bytes: image.bytes,
      width: image.width,
      height: image.height,
      language: normalizeLang(options.sourceLang) ?? '',
      region: this.options.region ?? 'US',
      timeZone: this.options.timeZone ?? 'America/New_York',
    })

    return this.limiter.run(async () => {
      for (let attempt = 0; attempt < MAX_REQUEST_ATTEMPTS; attempt += 1) {
        throwIfAborted(options.signal)
        const abort = requestAbortSignal(options.signal, this.requestTimeoutMs)
        try {
          const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-protobuf',
              'X-Goog-Api-Key': LENS_API_KEY,
              'Accept': '*/*',
            },
            body: request,
            signal: abort.signal,
          })
          if (!response.ok) throw lensHttpError(response.status)
          const parsed = parseLensResponse(new Uint8Array(await response.arrayBuffer()), [image.originalWidth, image.originalHeight])
          return {
            pageIndex: options.pageIndex,
            pageSize: [image.originalWidth, image.originalHeight],
            detectedLanguage: parsed.detectedLanguage,
            blocks: dedupeTextBlocks(parsed.blocks),
            timingMs: { recognize: Math.round(performance.now() - start) },
          }
        } catch (error) {
          if (options.signal?.aborted || attempt >= MAX_REQUEST_ATTEMPTS - 1 || !isRetryableLensError(error)) throw error
          await sleep(lensRetryDelayMs(attempt), options.signal)
        } finally {
          abort.cleanup()
        }
      }

      throw new Error('Lens OCR failed')
    })
  }
}

function lensHttpError(status: number): Error {
  const error = new Error(`Lens proxy failed: ${status}`) as Error & { status: number }
  error.status = status
  return error
}

function isRetryableLensError(error: unknown): boolean {
  const status = typeof (error as { status?: unknown } | null)?.status === 'number'
    ? (error as { status: number }).status
    : null
  if (status !== null) return status === 408 || status === 429 || status >= 500
  if (!(error instanceof Error)) return false
  return error.name === 'AbortError'
    || error.name === 'TypeError'
    || error.message === 'Failed to fetch'
    || error.message === 'Lens OCR timed out'
}

function lensRetryDelayMs(attempt: number): number {
  const base = RETRY_BASE_DELAY_MS * 2 ** attempt
  return base + Math.round(Math.random() * base * 0.25)
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  throwIfAborted(signal)
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup()
      resolve()
    }, ms)
    const onAbort = () => {
      cleanup()
      reject(signal?.reason instanceof Error ? signal.reason : new Error('operation aborted'))
    }
    const cleanup = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

function requestAbortSignal(parent: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; cleanup: () => void } {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('Lens OCR timed out')), timeoutMs)
  const onAbort = () => controller.abort(parent?.reason instanceof Error ? parent.reason : new Error('operation aborted'))
  if (parent?.aborted) onAbort()
  else parent?.addEventListener('abort', onAbort, { once: true })
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer)
      parent?.removeEventListener('abort', onAbort)
    },
  }
}

async function encodeImageForLens(image: ImagePixels): Promise<EncodedOcrImage> {
  const scale = Math.min(1, MAX_LENS_WIDTH / image.width, MAX_LENS_HEIGHT / image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  const source = document.createElement('canvas')
  source.width = image.width
  source.height = image.height
  const sourceCtx = source.getContext('2d')
  if (!sourceCtx) throw new Error('2d canvas unavailable')
  sourceCtx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)

  const out = document.createElement('canvas')
  out.width = width
  out.height = height
  const outCtx = out.getContext('2d')
  if (!outCtx) throw new Error('2d canvas unavailable')
  outCtx.drawImage(source, 0, 0, width, height)
  const blob = await new Promise<Blob>((resolve, reject) => {
    out.toBlob(value => value ? resolve(value) : reject(new Error('failed to encode image for Lens')), 'image/png')
  })
  return {
    bytes: new Uint8Array(await blob.arrayBuffer()),
    width,
    height,
    originalWidth: image.width,
    originalHeight: image.height,
  }
}

function createLensRequest(args: {
  bytes: Uint8Array
  width: number
  height: number
  language: string
  region: string
  timeZone: string
}): Uint8Array {
  const requestId = new lens.LensOverlayRequestId()
  // uuid is uint64 — pass a number, not a string like the old code did
  requestId.setUuid(Math.floor(Date.now() * 1000 + Math.random() * 1_000_000))
  requestId.setSequenceId(1)
  requestId.setImageSequenceId(1)

  const locale = new lens.LocaleContext()
  locale.setLanguage(args.language)
  locale.setRegion(args.region)
  locale.setTimeZone(args.timeZone)

  const filter = new lens.AppliedFilter()
  filter.setFilterType(lens.LensOverlayFilterType.AUTO_FILTER)
  const filters = new lens.AppliedFilters()
  filters.addFilter(filter)

  const client = new lens.LensOverlayClientContext()
  client.setPlatform(lens.Platform.PLATFORM_WEB)
  client.setSurface(lens.Surface.SURFACE_CHROMIUM)
  client.setLocaleContext(locale)
  client.setClientFilters(filters)

  const context = new lens.LensOverlayRequestContext()
  context.setRequestId(requestId)
  context.setClientContext(client)

  const metadata = new lens.ImageMetadata()
  metadata.setWidth(args.width)
  metadata.setHeight(args.height)
  const payload = new lens.ImagePayload()
  payload.setImageBytes(args.bytes)
  const imageData = new lens.ImageData()
  imageData.setPayload(payload)
  imageData.setImageMetadata(metadata)

  const objects = new lens.LensOverlayObjectsRequest()
  objects.setRequestContext(context)
  objects.setImageData(imageData)
  const request = new lens.LensOverlayServerRequest()
  request.setObjectsRequest(objects)
  return request.serializeBinary()
}

function parseLensResponse(bytes: Uint8Array, pageSize: readonly [number, number]): { detectedLanguage: string | null; blocks: TextBlock[] } {
  const response = lens.LensOverlayServerResponse.deserializeBinary(bytes)
  if (!response.hasObjectsResponse()) return { detectedLanguage: null, blocks: [] }
  const objects = response.getObjectsResponse()
  if (!objects.hasText()) return { detectedLanguage: null, blocks: [] }
  const text = objects.getText()
  if (!text.hasTextLayout()) return { detectedLanguage: text.getContentLanguage?.() || null, blocks: [] }

  const blocks: TextBlock[] = []
  for (const paragraph of text.getTextLayout().getParagraphsList()) {
    const lines = parseLines(paragraph.getLinesList(), pageSize)
    const sourceText = lines.map(line => line.text).filter(Boolean).join('\n').trim()
    if (!sourceText) continue
    const box = parseParagraphBox(paragraph, lines, pageSize)
    if (!box) continue
    blocks.push({
      bbox: box.bbox,
      polygon: box.polygon,
      text: sourceText,
      rotationDeg: box.rotationDeg,
      textDirection: paragraphDirection(paragraph, box.rotationDeg, box.bbox, lines, sourceText),
      confidence: 1,
      lines,
      words: lines.flatMap(line => line.words),
    })
  }
  return { detectedLanguage: text.getContentLanguage?.() || null, blocks }
}

function parseLines(lines: readonly any[], pageSize: readonly [number, number]): TextLine[] {
  const out: TextLine[] = []
  for (const line of lines) {
    const rawWords = line.getWordsList()
    const geometryWords = parseWords(rawWords, pageSize)
    const lineText = lensLineText(rawWords)
    if (!lineText) continue
    const box = parseLineBox(line, geometryWords, pageSize)
    if (!box) continue
    const words = parseWords(rawWords, pageSize, box.bbox)
    out.push({ bbox: box.bbox, text: lineText, rotationDeg: box.rotationDeg, words, oriented: box.oriented })
  }
  return out
}

function parseWords(words: readonly any[], pageSize: readonly [number, number], syntheticBBox?: BBox): TextWord[] {
  const out: TextWord[] = []
  for (const word of words) {
    const text = String(word.getPlainText?.() ?? '').trim()
    if (!text) continue
    const textSeparator = word.hasTextSeparator?.() ? String(word.getTextSeparator?.() ?? '') : undefined
    if (!word.hasGeometry()) {
      if (syntheticBBox && isMeaningfulPunctuationToken(text)) out.push({ bbox: syntheticBBox, text, textSeparator })
      continue
    }
    const geom = word.getGeometry()
    if (!geom.hasBoundingBox()) {
      if (syntheticBBox && isMeaningfulPunctuationToken(text)) out.push({ bbox: syntheticBBox, text, textSeparator })
      continue
    }
    const box = parseRotatedBox(geom.getBoundingBox(), pageSize)
    if (!box) {
      if (syntheticBBox && isMeaningfulPunctuationToken(text)) out.push({ bbox: syntheticBBox, text, textSeparator })
      continue
    }
    out.push({ bbox: box.bbox, text, textSeparator, oriented: box.oriented })
  }
  return out
}

function isMeaningfulPunctuationToken(text: string): boolean {
  const compact = text.replace(/\s+/gu, '')
  if (!compact || !/^[\p{P}\p{S}]+$/u.test(compact)) return false
  return /…|⋯|\.{2,}|。{2,}|[!?！？]{2,}|[—~〜]{2,}/u.test(compact)
}

function parseLineBox(line: any, words: readonly TextWord[], pageSize: readonly [number, number]): { bbox: BBox; polygon: Polygon; rotationDeg: number; oriented?: OrientedBox } | null {
  if (line.hasGeometry()) {
    const geom = line.getGeometry()
    if (geom.hasBoundingBox()) {
      const box = parseRotatedBox(geom.getBoundingBox(), pageSize)
      if (box) return box
    }
  }
  return boxFromWords(words)
}

function lensLineText(words: readonly any[]): string {
  let out = ''
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index]!
    const text = String(word.getPlainText?.() ?? '')
    const nextText = String(words[index + 1]?.getPlainText?.() ?? '')
    const separator = word.hasTextSeparator?.()
      ? String(word.getTextSeparator?.() ?? '')
      : inferredSeparator(text, nextText)
    out += text + separator
  }
  return out.replace(/[ \t]+/gu, ' ').trim()
}

function inferredSeparator(left: string, right: string): string {
  if (!left || !right) return ''
  const leftChars = [...left]
  const rightChars = [...right]
  const last = leftChars[leftChars.length - 1] ?? ''
  const first = rightChars[0] ?? ''
  if (isCjkLike(last) || isCjkLike(first)) return ''
  if (/^[\p{P}\p{S}]$/u.test(last) || /^[\p{P}\p{S}]$/u.test(first)) return ''
  return ' '
}

function isCjkLike(char: string): boolean {
  const cp = char.codePointAt(0)
  if (cp === undefined) return false
  return (cp >= 0x3040 && cp <= 0x30FF)
    || (cp >= 0x3400 && cp <= 0x4DBF)
    || (cp >= 0x4E00 && cp <= 0x9FFF)
    || (cp >= 0xF900 && cp <= 0xFAFF)
}

function parseParagraphBox(paragraph: any, lines: readonly TextLine[], pageSize: readonly [number, number]): { bbox: BBox; polygon: Polygon; rotationDeg: number; oriented?: OrientedBox } | null {
  if (paragraph.hasGeometry()) {
    const geom = paragraph.getGeometry()
    if (geom.hasBoundingBox()) return parseRotatedBox(geom.getBoundingBox(), pageSize)
  }
  return boxFromLines(lines)
}

function boxFromLines(lines: readonly TextLine[]): { bbox: BBox; polygon: Polygon; rotationDeg: number; oriented?: OrientedBox } | null {
  const bbox = unionBBoxes(lines.map(line => line.bbox))
  if (!bbox) return null
  return { bbox, polygon: bboxToPolygon(bbox), rotationDeg: maxAbsRotation(lines) }
}

function boxFromWords(words: readonly TextWord[]): { bbox: BBox; polygon: Polygon; rotationDeg: number; oriented?: OrientedBox } | null {
  const bbox = unionBBoxes(words.map(word => word.bbox))
  if (!bbox) return null
  return { bbox, polygon: bboxToPolygon(bbox), rotationDeg: 0 }
}

function parseRotatedBox(box: any, pageSize: readonly [number, number]): { bbox: BBox; polygon: Polygon; rotationDeg: number; oriented: OrientedBox } | null {
  if (box.getCoordinateType() !== lens.CoordinateType.NORMALIZED) return null
  const [pageW, pageH] = pageSize
  const cx = box.getCenterX() * pageW
  const cy = box.getCenterY() * pageH
  const w = box.getWidth() * pageW
  const h = box.getHeight() * pageH
  const rotationDeg = (box.getRotationZ() || 0) * 180 / Math.PI
  const polygon = orientedRect(cx, cy, w, h, rotationDeg)
  return { bbox: polygonBBox(polygon, pageSize), polygon, rotationDeg, oriented: { cx, cy, w, h, rotationDeg } }
}

function dedupeTextBlocks(blocks: readonly TextBlock[]): TextBlock[] {
  const kept: TextBlock[] = []
  for (const block of blocks) {
    const duplicateIndex = kept.findIndex(existing => sameTextRegion(block, existing))
    if (duplicateIndex === -1) kept.push(block)
    else if (blockQuality(block) > blockQuality(kept[duplicateIndex]!)) kept[duplicateIndex] = block
  }
  return kept
}

function sameTextRegion(a: TextBlock, b: TextBlock): boolean {
  if (iou(a.bbox, b.bbox) >= 0.90) return true
  if (a.lines.length && a.lines.length === b.lines.length) {
    const rightLines = sortBBoxes(b.lines.map(line => line.bbox))
    const overlaps = sortBBoxes(a.lines.map(line => line.bbox)).map((box, index) => iou(box, rightLines[index]!))
    return overlaps.length > 0 && Math.min(...overlaps) >= 0.70
  }
  return false
}

function blockQuality(block: TextBlock): number {
  return block.lines.length * 10 + block.words.length + block.text.length / 100
}

function paragraphDirection(
  paragraph: any,
  rotationDeg: number,
  bbox: BBox,
  lines: readonly TextLine[],
  text: string,
): TextDirection {
  const direction = paragraph.getWritingDirection?.() ?? 0
  if (direction === 2 || Math.abs(rotationDeg) > 45) return 'vertical'
  if (looksLikeVerticalCjk(text, bbox, lines)) return 'vertical'
  return 'horizontal'
}

function looksLikeVerticalCjk(text: string, bbox: BBox, lines: readonly TextLine[]): boolean {
  if (!hasCjkText(text)) return false
  const boxes = lines.length ? lines.map(line => line.bbox) : [bbox]
  const meaningful = boxes.filter(box => bboxWidth(box) >= 2 && bboxHeight(box) >= 8)
  if (!meaningful.length) return false
  const verticalLines = meaningful.filter(isTallNarrowBox).length
  return verticalLines > 0 && verticalLines * 2 >= meaningful.length
}

function isTallNarrowBox(bbox: BBox): boolean {
  const width = bboxWidth(bbox)
  const height = bboxHeight(bbox)
  return height >= width * 2.0 && width <= 96
}

function hasCjkText(text: string): boolean {
  for (const char of text) {
    const cp = char.codePointAt(0)
    if (cp === undefined) continue
    if (cp >= 0x3040 && cp <= 0x30FF) return true
    if (cp >= 0x3400 && cp <= 0x4DBF) return true
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true
    if (cp >= 0xF900 && cp <= 0xFAFF) return true
  }
  return false
}

function bboxWidth(bbox: BBox): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxHeight(bbox: BBox): number {
  return Math.max(0, bbox[3] - bbox[1])
}

function normalizeLang(lang: string | null): string | null {
  if (!lang) return null
  const lower = lang.toLowerCase()
  if (lower.startsWith('zh-hant') || lower === 'zh-tw') return 'zh-Hant'
  if (lower.startsWith('zh')) return 'zh-Hans'
  return lower.split('-')[0] ?? lower
}

function orientedRect(cx: number, cy: number, w: number, h: number, rotationDeg: number): Polygon {
  const rad = rotationDeg * Math.PI / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([x, y]) => [cx + x * cos - y * sin, cy + x * sin + y * cos] as Point)
}

function polygonBBox(polygon: Polygon, pageSize: readonly [number, number]): BBox {
  const xs = polygon.map(p => p[0])
  const ys = polygon.map(p => p[1])
  return clipBBox([Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys))], pageSize)
}

function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox {
  return [Math.max(0, bbox[0]), Math.max(0, bbox[1]), Math.min(pageSize[0], bbox[2]), Math.min(pageSize[1], bbox[3])]
}

function bboxToPolygon(b: BBox): Polygon {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

function unionBBoxes(boxes: readonly BBox[]): BBox | null {
  if (!boxes.length) return null
  return [Math.min(...boxes.map(b => b[0])), Math.min(...boxes.map(b => b[1])), Math.max(...boxes.map(b => b[2])), Math.max(...boxes.map(b => b[3]))]
}

function maxAbsRotation(lines: readonly TextLine[]): number {
  if (!lines.length) return 0
  return lines.reduce((best, line) => Math.abs(line.rotationDeg) > Math.abs(best) ? line.rotationDeg : best, 0)
}

function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  return inter / (area(a) + area(b) - inter)
}

function area(b: BBox): number {
  return Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
}

function sortBBoxes(boxes: readonly BBox[]): BBox[] {
  return [...boxes].sort((a, b) => a[1] - b[1] || a[0] - b[0])
}
