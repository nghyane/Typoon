import type { BBox, Point, Polygon } from '../../domain/geometry'
import type { ImagePixels } from '../../domain/image'
import type { TextBlock, TextLine, RecognizedTextPage, TextWord, TextDirection } from '../../domain/text'
import type { TextRecognizer, TextRecognitionOptions } from '../text'
import * as lens from 'chrome-lens-ocr/src/utils/proto_generated/lens_overlay_server_pb.cjs'

const DEFAULT_ENDPOINT = 'https://927251094806098001.discordsays.com/lens/v1/crupload'
const LENS_API_KEY = 'AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY'
const MAX_LENS_DIMENSION = 1500

export interface LensTextRecognizerOptions {
  readonly endpoint?: string
  readonly region?: string
  readonly timeZone?: string
}

export class LensTextRecognizer implements TextRecognizer {
  readonly name = 'lens-text-recognizer'
  private readonly endpoint: string

  constructor(private readonly options: LensTextRecognizerOptions = {}) {
    this.endpoint = options.endpoint ?? DEFAULT_ENDPOINT
  }

  async recognizeText(image: ImagePixels, options: TextRecognitionOptions): Promise<RecognizedTextPage> {
    const start = performance.now()
    const encoded = await encodeImageForLens(image)
    const request = createLensRequest({
      bytes: encoded.bytes,
      width: encoded.width,
      height: encoded.height,
      language: normalizeLang(options.sourceLang) ?? '',
      region: this.options.region ?? 'US',
      timeZone: this.options.timeZone ?? 'America/New_York',
    })
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-protobuf',
        'X-Goog-Api-Key': LENS_API_KEY,
        'Accept': '*/*',
      },
      body: request,
      signal: options.signal,
    })
    if (!response.ok) throw new Error(`Lens proxy failed: ${response.status}`)

    const parsed = parseLensResponse(new Uint8Array(await response.arrayBuffer()), [image.width, image.height])
    return {
      pageIndex: options.pageIndex,
      pageSize: [image.width, image.height],
      detectedLanguage: parsed.detectedLanguage,
      blocks: dedupeTextBlocks(parsed.blocks),
      timingMs: { recognize: Math.round(performance.now() - start) },
    }
  }
}

async function encodeImageForLens(image: ImagePixels): Promise<{ bytes: Uint8Array; width: number; height: number }> {
  const scale = Math.min(1, MAX_LENS_DIMENSION / Math.max(image.width, image.height))
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
  return { bytes: new Uint8Array(await blob.arrayBuffer()), width, height }
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
  requestId.setUuid(String(Date.now()) + String(Math.floor(Math.random() * 1_000_000)))
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
      textDirection: paragraphDirection(paragraph, box.rotationDeg),
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
    const words = parseWords(line.getWordsList(), pageSize)
    const lineText = words.map((word, index) => word.text + (word.textSeparator ?? (index === words.length - 1 ? '' : ' '))).join('').replace(/[ \t]+/gu, ' ').trim()
    if (!lineText || !line.hasGeometry()) continue
    const geom = line.getGeometry()
    if (!geom.hasBoundingBox()) continue
    const box = parseRotatedBox(geom.getBoundingBox(), pageSize)
    if (!box) continue
    out.push({ bbox: box.bbox, text: lineText, rotationDeg: box.rotationDeg, words })
  }
  return out
}

function parseWords(words: readonly any[], pageSize: readonly [number, number]): TextWord[] {
  const out: TextWord[] = []
  for (const word of words) {
    const text = String(word.getPlainText?.() ?? '').trim()
    if (!text || !word.hasGeometry()) continue
    const geom = word.getGeometry()
    if (!geom.hasBoundingBox()) continue
    const box = parseRotatedBox(geom.getBoundingBox(), pageSize)
    if (!box) continue
    const textSeparator = word.hasTextSeparator?.() ? String(word.getTextSeparator?.() ?? '') : undefined
    out.push({ bbox: box.bbox, text, textSeparator })
  }
  return out
}

function parseParagraphBox(paragraph: any, lines: readonly TextLine[], pageSize: readonly [number, number]): { bbox: BBox; polygon: Polygon; rotationDeg: number } | null {
  if (paragraph.hasGeometry()) {
    const geom = paragraph.getGeometry()
    if (geom.hasBoundingBox()) return parseRotatedBox(geom.getBoundingBox(), pageSize)
  }
  return boxFromLines(lines)
}

function boxFromLines(lines: readonly TextLine[]): { bbox: BBox; polygon: Polygon; rotationDeg: number } | null {
  const bbox = unionBBoxes(lines.map(line => line.bbox))
  if (!bbox) return null
  return { bbox, polygon: bboxToPolygon(bbox), rotationDeg: maxAbsRotation(lines) }
}

function parseRotatedBox(box: any, pageSize: readonly [number, number]): { bbox: BBox; polygon: Polygon; rotationDeg: number } | null {
  if (box.getCoordinateType() !== lens.CoordinateType.NORMALIZED) return null
  const [pageW, pageH] = pageSize
  const cx = box.getCenterX() * pageW
  const cy = box.getCenterY() * pageH
  const w = box.getWidth() * pageW
  const h = box.getHeight() * pageH
  const rotationDeg = (box.getRotationZ() || 0) * 180 / Math.PI
  const polygon = orientedRect(cx, cy, w, h, rotationDeg)
  return { bbox: polygonBBox(polygon, pageSize), polygon, rotationDeg }
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

function paragraphDirection(paragraph: any, rotationDeg: number): TextDirection {
  const direction = paragraph.getWritingDirection?.() ?? 0
  if (direction === 2 || Math.abs(rotationDeg) > 45) return 'vertical'
  return 'horizontal'
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
