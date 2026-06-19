import type { TextRole } from '../domain/planning'
import type { BubbleShapeProfile } from './bubbleShape'
import type { FontProfile } from './font'
import type { DomMeasurer } from './textMeasure'
import { canBreakTokenPerCharacter } from './textScript'

export type LineLayoutCandidate = 'baseline' | 'vertical'

const MAX_COMPOSE_LINES = 9
const OVERFLOW_PENALTY = 10_000

interface TextSegment {
  readonly text: string
  readonly joinBefore: '' | ' '
}

export interface LineComposition {
  readonly text: string
  readonly lines: readonly string[]
  readonly candidate: LineLayoutCandidate
  readonly lineCount: number
  readonly heightPx: number
  readonly overflowHeight: boolean
  readonly overflowWidth: boolean
  readonly fits: boolean
  readonly score: number
  /** Max ratio of line width / line limit across all lines. >0.85 = too dense. */
  readonly maxFill: number
}

export function composeLines(args: {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly fontSizePx: number
  readonly font: FontProfile
  readonly fontWeight: string
  readonly role: TextRole
  readonly direction?: 'horizontal' | 'vertical'
  readonly shapeProfile?: BubbleShapeProfile
  readonly sourceLineCount?: number
  readonly measurer: DomMeasurer
}): LineComposition {
  if (args.direction === 'vertical') return composeVerticalLines(args)

  const raw = args.text
  const segments = horizontalSegments(raw, args.width, args.fontSizePx, args.fontWeight, args.measurer)

  const lineHeightPx = args.fontSizePx * args.font.lineHeightRatio
  const maxLinesByHeight = Math.max(1, Math.floor(args.height / Math.max(1, lineHeightPx)))
  const maxLines = Math.min(segments.length || 1, MAX_COMPOSE_LINES, maxLinesByHeight)
  if (!segments.length) return emptyComposition(args.text)
  const widths = segmentWidths(segments, args.fontSizePx, args.fontWeight, args.measurer)

  let best: LineComposition | null = null
  for (let lineCount = 1; lineCount <= maxLines; lineCount += 1) {
    const candidate = composeForLineCount({ ...args, segments, widths, lineCount, lineHeightPx })
    if (!best || candidate.score < best.score) best = candidate
  }

  return best ?? fallbackComposition(args.text, lineHeightPx, args.height)
}

function horizontalSegments(
  raw: string,
  width: number,
  fontSizePx: number,
  fontWeight: string,
  measurer: DomMeasurer,
): TextSegment[] {
  const words = raw.split(/\s+/u).filter(Boolean)
  const charCount = [...raw].filter(ch => !/\s/u.test(ch)).length

  if (words.length <= 1 && charCount > 4 && canBreakTokenPerCharacter(raw)) {
    return [...(words[0] ?? '')].filter(ch => !/\s/u.test(ch)).map(text => ({ text, joinBefore: '' }))
  }

  const segments: TextSegment[] = []
  for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
    const pieces = splitTokenForFit(words[wordIndex]!, width, fontSizePx, fontWeight, measurer)
    for (let pieceIndex = 0; pieceIndex < pieces.length; pieceIndex += 1) {
      segments.push({
        text: pieces[pieceIndex]!,
        joinBefore: wordIndex > 0 && pieceIndex === 0 ? ' ' : '',
      })
    }
  }
  return segments
}

function splitTokenForFit(
  token: string,
  width: number,
  fontSizePx: number,
  fontWeight: string,
  measurer: DomMeasurer,
): readonly string[] {
  const chars = [...token].filter(ch => !/\s/u.test(ch))
  if (chars.length <= 1) return [token]
  const tokenWidth = measurer.textWidth({ text: token, fontSizePx, fontWeight })
  if (tokenWidth <= width + 0.5) return [token]
  if (canBreakTokenPerCharacter(token)) return chars
  if (!hasWordText(token)) return chars
  return splitWordTokenPunctuation(token)
}

function splitWordTokenPunctuation(token: string): readonly string[] {
  const out: string[] = []
  let current = ''
  const chars = [...token]
  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i]!
    if (isWordChar(char)) {
      current += char
      continue
    }

    let end = i + 1
    while (end < chars.length && !isWordChar(chars[end]!)) end += 1
    const run = chars.slice(i, end)
    if (run.length >= 3) {
      if (current) {
        out.push(current)
        current = ''
      }
      out.push(...run)
    } else {
      current += run.join('')
    }
    i = end - 1
  }
  if (current) out.push(current)
  return out.length > 1 ? out : [token]
}

function composeVerticalLines(args: {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly fontSizePx: number
  readonly font: FontProfile
  readonly role: TextRole
  readonly sourceLineCount?: number
}): LineComposition {
  const chars = [...args.text].filter(char => !/\s/u.test(char))
  const lineHeightPx = args.fontSizePx * args.font.lineHeightRatio
  if (!chars.length) return emptyComposition(args.text)

  const charsPerColumn = Math.max(1, Math.floor(args.height / Math.max(1, args.fontSizePx)))
  const maxColumnsByWidth = Math.max(1, Math.floor(args.width / Math.max(1, lineHeightPx)))
  const requiredColumns = Math.ceil(chars.length / charsPerColumn)
  const sourceColumns = args.sourceLineCount ?? requiredColumns
  const columnCount = Math.max(1, Math.min(requiredColumns, MAX_COMPOSE_LINES))
  const lines = chunkChars(chars, Math.ceil(chars.length / columnCount))
  const usedWidth = lines.length * lineHeightPx
  const usedHeight = Math.max(...lines.map(line => [...line].length), 0) * args.fontSizePx
  const overflowWidth = usedWidth > args.width + 0.5 || requiredColumns > maxColumnsByWidth
  const overflowHeight = usedHeight > args.height + 0.5
  const columnPenalty = Math.abs(lines.length - sourceColumns) * 4

  return {
    text: lines.join('\n'),
    lines,
    candidate: 'vertical',
    lineCount: lines.length,
    heightPx: usedHeight,
    overflowHeight,
    overflowWidth,
    fits: !overflowWidth && !overflowHeight,
    score: columnPenalty + (overflowWidth || overflowHeight ? OVERFLOW_PENALTY : 0),
    maxFill: usedHeight / Math.max(1, args.height),
  }
}

function chunkChars(chars: readonly string[], chunkSize: number): string[] {
  const out: string[] = []
  for (let i = 0; i < chars.length; i += chunkSize) out.push(chars.slice(i, i + chunkSize).join(''))
  return out
}

function composeForLineCount(args: {
  readonly text: string
  readonly segments: readonly TextSegment[]
  readonly widths: readonly (readonly number[])[]
  readonly lineCount: number
  readonly lineHeightPx: number
  readonly width: number
  readonly height: number
  readonly fontSizePx: number
  readonly font: FontProfile
  readonly fontWeight: string
  readonly role: TextRole
  readonly shapeProfile?: BubbleShapeProfile
  readonly sourceLineCount?: number
  readonly measurer: DomMeasurer
}): LineComposition {
  const n = args.segments.length
  const dp: number[][] = Array.from({ length: args.lineCount + 1 }, () => Array(n + 1).fill(Number.POSITIVE_INFINITY))
  const prev: number[][] = Array.from({ length: args.lineCount + 1 }, () => Array(n + 1).fill(-1))
  dp[0]![0] = 0

  for (let line = 1; line <= args.lineCount; line += 1) {
    for (let end = line; end <= n; end += 1) {
      for (let start = line - 1; start < end; start += 1) {
        const before = dp[line - 1]![start]!
        if (!Number.isFinite(before)) continue
        const cost = before + lineCost(args, start, end, line - 1)
        if (cost < dp[line]![end]!) {
          dp[line]![end] = cost
          prev[line]![end] = start
        }
      }
    }
  }

  const lines = recoverLines(args.segments, prev, args.lineCount, n)
  const lineWidths = recoverWidths(args.widths, prev, args.lineCount, n)
  const overflowWidth = lineWidths.some((w, index) => {
    const limit = lineWidthLimit(args.width, args.shapeProfile, args.lineCount, lines.length, index)
    return w > limit + 0.5
  })
  const heightPx = lines.length * args.lineHeightPx
  const overflowHeight = heightPx > args.height + 0.5
  const lineCountPenalty = Math.abs(lines.length - (args.sourceLineCount ?? lines.length)) * 4
  const score = (dp[args.lineCount]![n] ?? Number.POSITIVE_INFINITY) + lineCountPenalty
  const maxFill = lineWidths.reduce((max, w, i) => {
    const limit = lineWidthLimit(args.width, args.shapeProfile, args.lineCount, lines.length, i)
    return Math.max(max, w / Math.max(1, limit))
  }, 0)

  return {
    text: lines.join('\n'),
    lines,
    candidate: 'baseline',
    lineCount: lines.length,
    heightPx,
    overflowHeight,
    overflowWidth,
    fits: !overflowWidth && !overflowHeight,
    score,
    maxFill,
  }
}

function lineWidthLimit(
  maxWidth: number,
  shapeProfile: BubbleShapeProfile | undefined,
  requestedLineCount: number,
  actualLineCount: number,
  lineIndex: number,
): number {
  if (!shapeProfile) return maxWidth
  if (shapeProfile.kind === 'rect') return maxWidth
  // Map actual line index to the expected position in the requested line count
  const mappedIndex = actualLineCount <= 1 ? 0
    : Math.round((lineIndex / (actualLineCount - 1)) * (requestedLineCount - 1))
  return Math.min(maxWidth, shapeProfile.widthAt(mappedIndex, requestedLineCount))
}

function lineCost(args: {
  readonly segments: readonly TextSegment[]
  readonly widths: readonly (readonly number[])[]
  readonly lineCount: number
  readonly width: number
  readonly shapeProfile?: BubbleShapeProfile
}, start: number, end: number, lineIndex: number): number {
  const actualWidth = args.widths[start]?.[end] ?? Number.POSITIVE_INFINITY
  const lineLimit = lineWidthLimit(args.width, args.shapeProfile, args.lineCount, args.lineCount, lineIndex)

  // Hard constraint: overflow is never acceptable.
  if (actualWidth > lineLimit) return OVERFLOW_PENALTY

  // Cost = wasted horizontal space. Fuller lines → lower cost.
  const fill = actualWidth / Math.max(1, lineLimit)
  let cost = Math.pow(1 - fill, 2) * 100

  const line = segmentText(args.segments, start, end)
  const firstWord = args.segments[start]?.text ?? ''
  const lastWord = args.segments[end - 1]?.text ?? ''

  if (args.lineCount > 1 && end - start === 1 && isShortLatinWord(line)) cost += 80

  // Pro typesetting rule #1: break after punctuation (comma, question, exclamation, semicolon, colon, em-dash).
  if (/[,.!?;:—…。，、！？；：]$/u.test(line)) cost -= 6

  // Pro rule: never start a line with punctuation (bad break).
  if (/^[,.!?;:—…。，、！？；：]/u.test(firstWord)) cost += 30

  // Slight bonus for not breaking between a short word and what follows.
  if (end < args.segments.length && [...lastWord].length <= 2) cost += 4

  return cost
}

function recoverLines(segments: readonly TextSegment[], prev: readonly number[][], lineCount: number, n: number): string[] {
  const ranges: Array<readonly [number, number]> = []
  let end = n
  for (let line = lineCount; line > 0; line -= 1) {
    const start = prev[line]?.[end] ?? -1
    if (start < 0) return [segmentText(segments, 0, segments.length)]
    ranges.push([start, end])
    end = start
  }
  return ranges.reverse().map(([start, finish]) => segmentText(segments, start, finish))
}

function segmentWidths(segments: readonly TextSegment[], fontSizePx: number, fontWeight: string, measurer: DomMeasurer): number[][] {
  return segments.map((_, start) => {
    const row = Array(segments.length + 1).fill(Number.POSITIVE_INFINITY)
    for (let end = start + 1; end <= segments.length; end += 1) {
      row[end] = measurer.textWidth({ text: segmentText(segments, start, end), fontSizePx, fontWeight })
    }
    return row
  })
}

function segmentText(segments: readonly TextSegment[], start: number, end: number): string {
  let out = ''
  for (let i = start; i < end; i += 1) {
    const segment = segments[i]
    if (!segment) continue
    out += out ? `${segment.joinBefore}${segment.text}` : segment.text
  }
  return out
}

function isShortLatinWord(text: string): boolean {
  const letters = [...text].filter(char => /\p{L}/u.test(char)).join('')
  if (!letters || canBreakTokenPerCharacter(letters)) return false
  return [...letters].length <= 3
}

function hasWordText(text: string): boolean {
  return [...text].some(isWordChar)
}

function isWordChar(char: string): boolean {
  return /[\p{L}\p{N}\p{M}]/u.test(char)
}

function recoverWidths(widths: readonly (readonly number[])[], prev: readonly number[][], lineCount: number, n: number): number[] {
  const out: number[] = []
  let end = n
  for (let line = lineCount; line > 0; line -= 1) {
    const start = prev[line]?.[end] ?? -1
    if (start < 0) return [widths[0]?.[n] ?? Number.POSITIVE_INFINITY]
    out.push(widths[start]?.[end] ?? Number.POSITIVE_INFINITY)
    end = start
  }
  return out.reverse()
}

function emptyComposition(text: string): LineComposition {
  return { text, lines: [], candidate: 'baseline', lineCount: 0, heightPx: 0, overflowHeight: false, overflowWidth: false, fits: true, score: 0, maxFill: 0 }
}

function fallbackComposition(text: string, lineHeightPx: number, height: number): LineComposition {
  return { text, lines: [text], candidate: 'baseline', lineCount: 1, heightPx: lineHeightPx, overflowHeight: lineHeightPx > height, overflowWidth: true, fits: false, score: Number.POSITIVE_INFINITY, maxFill: 1 }
}
