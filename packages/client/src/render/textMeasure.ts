import type { FontProfile } from './font'

export interface MeasureRequest {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly fontSizePx: number
  readonly fontWeight: string
}

export interface BlockMeasureRequest {
  readonly text: string
  readonly width: number
  readonly fontSizePx: number
  readonly fontWeight: string
}

export interface TextWidthRequest {
  readonly text: string
  readonly fontSizePx: number
  readonly fontWeight: string
}

export interface TextBlockMeasure {
  readonly widthPx: number
  readonly heightPx: number
  readonly lineCount: number
  readonly overflowWidth: boolean
}

export interface DomMeasurer {
  fits(request: MeasureRequest): boolean
  measure(request: BlockMeasureRequest): TextBlockMeasure
  textWidth(request: TextWidthRequest): number
  destroy(): void
}

export function createDomMeasurer(font: FontProfile): DomMeasurer {
  const el = document.createElement('div')
  el.style.position = 'fixed'
  el.style.left = '-10000px'
  el.style.top = '0'
  el.style.visibility = 'hidden'
  el.style.pointerEvents = 'none'
  el.style.boxSizing = 'border-box'
  el.style.display = 'block'
  el.style.textAlign = 'center'
  el.style.whiteSpace = 'normal'
  el.style.overflowWrap = 'normal'
  el.style.wordBreak = 'normal'
  el.style.fontFamily = font.cssFamily
  el.style.padding = '0'
  document.body.appendChild(el)

  return {
    fits(request: MeasureRequest): boolean {
      el.style.display = 'block'
      el.style.width = `${request.width}px`
      el.style.height = 'auto'
      el.style.fontSize = `${request.fontSizePx}px`
      el.style.lineHeight = `${request.fontSizePx * font.lineHeightRatio}px`
      el.style.fontWeight = request.fontWeight
      el.style.whiteSpace = 'pre'
      el.textContent = request.text

      const rect = el.getBoundingClientRect()
      return el.scrollWidth <= Math.ceil(request.width) && rect.height <= request.height + 0.5
    },
    measure(request: BlockMeasureRequest): TextBlockMeasure {
      el.style.display = 'block'
      el.style.width = `${request.width}px`
      el.style.height = 'auto'
      el.style.fontSize = `${request.fontSizePx}px`
      el.style.lineHeight = `${request.fontSizePx * font.lineHeightRatio}px`
      el.style.fontWeight = request.fontWeight
      el.style.whiteSpace = 'pre-wrap'
      el.style.overflowWrap = 'break-word'
      el.style.wordBreak = 'normal'
      el.textContent = request.text

      const rect = el.getBoundingClientRect()
      const lineHeightPx = request.fontSizePx * font.lineHeightRatio

      return {
        widthPx: rect.width,
        heightPx: rect.height,
        lineCount: Math.max(1, Math.round(rect.height / lineHeightPx)),
        overflowWidth: el.scrollWidth > Math.ceil(request.width),
      }
    },
    textWidth(request: TextWidthRequest): number {
      el.style.display = 'inline-block'
      el.style.width = 'auto'
      el.style.height = 'auto'
      el.style.fontSize = `${request.fontSizePx}px`
      el.style.lineHeight = `${request.fontSizePx * font.lineHeightRatio}px`
      el.style.fontWeight = request.fontWeight
      el.style.whiteSpace = 'nowrap'
      el.textContent = request.text

      return el.getBoundingClientRect().width
    },
    destroy(): void {
      el.remove()
    },
  }
}

export function maxFittingSize(args: {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly hiBound: number
  readonly fontWeight: string
  readonly measurer: DomMeasurer
  readonly minFontSize: number
}): number {
  let lo = args.minFontSize
  let hi = args.hiBound
  let best = args.minFontSize
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (args.measurer.fits({
      text: args.text,
      width: args.width,
      height: args.height,
      fontSizePx: mid,
      fontWeight: args.fontWeight,
    })) {
      best = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return best
}
