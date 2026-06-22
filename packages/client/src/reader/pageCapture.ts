// reader/pageCapture.ts — build the OCR canvas for one page: core + halo.
//
// Canvas layout top→bottom: [prev bottom strip][page N][next top strip].
// Invariant (I9): canvas width === page N source width * captureScale, so page
// N is drawn at x=0 and canvas→page-N-source mapping needs no x offset. Neighbor
// strips are centered (their x offset is irrelevant: halo-only blocks are
// dropped by centroid-in-core, and seam blocks are owned by page N at x=0).

import type { EncodedOcrImage } from '../recognizers/text'
import type { ImagePixels } from '../domain/image'
import type { PageScanUnit } from '../domain/pageScan'
import type { PageSize } from '../domain/source'
import type { LoadedPage } from './pageProvider'
import type { ScanConfig } from './translationConfig'

export interface CapturedPageScan {
  readonly encoded: EncodedOcrImage
  readonly image: ImagePixels
  readonly captureScale: number    // canvas px = source px * captureScale
  readonly haloTopPx: number       // page N source px
  readonly source: PageSize        // page N
}

export async function capturePageScan(
  unit: PageScanUnit,
  loadPage: (index: number) => Promise<LoadedPage>,
  config: ScanConfig,
  signal: AbortSignal,
): Promise<CapturedPageScan> {
  const captureScale = Math.min(1, config.maxCaptureWidth / Math.max(1, unit.source.width))
  const srcHeight = unit.haloTopPx + unit.source.height + unit.haloBottomPx
  const cw = Math.max(1, Math.round(unit.source.width * captureScale))
  const ch = Math.max(1, Math.round(srcHeight * captureScale))

  const canvas = document.createElement('canvas')
  canvas.width = cw
  canvas.height = ch
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, cw, ch)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  // core: page N at y = haloTopPx
  await drawWholePage(ctx, loadPage, unit.pageIndex, unit.haloTopPx, captureScale, cw, signal)
  // top halo: bottom strip of prev page → top of canvas
  if (unit.prevIndex !== null && unit.haloTopPx > 0) {
    await drawNeighborStrip(ctx, loadPage, unit.prevIndex, 'bottom', unit.haloTopPx, 0, captureScale, cw, signal)
  }
  // bottom halo: top strip of next page → below core
  if (unit.nextIndex !== null && unit.haloBottomPx > 0) {
    const destSrcY = unit.haloTopPx + unit.source.height
    await drawNeighborStrip(ctx, loadPage, unit.nextIndex, 'top', unit.haloBottomPx, destSrcY, captureScale, cw, signal)
  }

  const pixels = ctx.getImageData(0, 0, cw, ch)
  const blob = await canvasToOcrBlob(canvas)
  return {
    encoded: {
      bytes: new Uint8Array(await blob.arrayBuffer()),
      width: cw,
      height: ch,
      originalWidth: cw,
      originalHeight: ch,
    },
    image: { width: cw, height: ch, data: pixels.data },
    captureScale,
    haloTopPx: unit.haloTopPx,
    source: unit.source,
  }
}

async function drawWholePage(
  ctx: CanvasRenderingContext2D,
  loadPage: (index: number) => Promise<LoadedPage>,
  index: number,
  destSrcY: number,
  scale: number,
  canvasWidth: number,
  signal: AbortSignal,
): Promise<void> {
  throwIfAborted(signal)
  const page = await loadPage(index)
  const bitmap = await createImageBitmap(page.blob)
  try {
    const dw = page.size.width * scale
    ctx.drawImage(bitmap, centerOffset(canvasWidth, dw), destSrcY * scale, dw, page.size.height * scale)
  } finally {
    bitmap.close()
  }
}

async function drawNeighborStrip(
  ctx: CanvasRenderingContext2D,
  loadPage: (index: number) => Promise<LoadedPage>,
  index: number,
  edge: 'top' | 'bottom',
  stripSrcHeight: number,
  destSrcY: number,
  scale: number,
  canvasWidth: number,
  signal: AbortSignal,
): Promise<void> {
  throwIfAborted(signal)
  const page = await loadPage(index)
  const bitmap = await createImageBitmap(page.blob)
  try {
    const sh = Math.min(stripSrcHeight, page.size.height)
    const sy = edge === 'bottom' ? page.size.height - sh : 0
    const dw = page.size.width * scale
    ctx.drawImage(bitmap, 0, sy, page.size.width, sh, centerOffset(canvasWidth, dw), destSrcY * scale, dw, sh * scale)
  } finally {
    bitmap.close()
  }
}

function centerOffset(canvasWidth: number, drawWidth: number): number {
  return (canvasWidth - drawWidth) / 2
}

function canvasToOcrBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('failed to encode page scan')), 'image/jpeg', 0.92)
  })
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
