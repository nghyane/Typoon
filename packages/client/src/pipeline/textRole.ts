import type { TextRole } from '../domain/planning'
import type { TextBlock } from '../domain/text'

/**
 * SFX classification uses two independent signals that do not depend on ONNX
 * region detection or pixel access — only OCR block data is available here.
 *
 * Signal A — Katakana SFX (Japanese):
 *   Japanese onomatopoeia is almost exclusively written in katakana.  When a
 *   short block is katakana-dominant and visually distinct from body text it
 *   is confidently SFX.
 *
 * Signal B — Visual SFX (any script):
 *   For Hangul, Hanzi, Hiragana, and Latin scripts the character set alone
 *   cannot distinguish SFX from regular text.  We require extreme visual
 *   evidence: very few characters AND dramatically larger than body text.
 *   This avoids false positives on headings, emphasis, and short dialogue
 *   lines that happen to be somewhat larger than average.
 *
 * Signal C — repeated glyph SFX:
 *   OCR often returns footsteps/knocks as the same glyph repeated across
 *   multiple lines (e.g. 嗒\n嗒\n嗒).  These are sound effects even when their
 *   font is not a huge page outlier.
 *
 * OCR rotation can be noisy inside speech bubbles, so angle alone is not a
 * role signal.  Rotated text still renders with its OCR angle when another
 * signal classifies it as SFX.
 */
const KATAKANA_MAX_CHARS = 6
const KATAKANA_ASPECT = 1.5
const KATAKANA_FONT_RATIO = 1.5

const VISUAL_MAX_CHARS = 3
const VISUAL_ASPECT = 2.5
const VISUAL_FONT_RATIO = 2.5
const VISUAL_EXTREME_FONT_RATIO = 3.2
const REPEATED_GLYPH_MIN_CHARS = 3
const REPEATED_GLYPH_MAX_CHARS = 8

export interface TextRoleContext {
  /** Median font size across the page — the "body" text size. */
  readonly bodyFontPx?: number
}

export function textRoleContext(blocks: readonly TextBlock[]): TextRoleContext {
  const samples = blocks
    .map(blockSourceFontPx)
    .filter(sample => sample > 0)
    .sort((a, b) => a - b)

  if (!samples.length) return {}

  const bodyFontPx = samples[Math.floor((samples.length - 1) / 2)]!
  return { bodyFontPx }
}

export function classifyTextBlockRole(block: TextBlock, context: TextRoleContext = {}): TextRole {
  const chars = [...block.text].filter(ch => !/\s/u.test(ch)).length

  // Long blocks are narration.
  if (chars > 30) return 'narration'

  const blockFont = blockSourceFontPx(block)
  const bodyFont = context.bodyFontPx ?? 0

  // Signal A: katakana-dominant → Japanese SFX
  if (isKatakanaSfx(block, chars, blockFont, bodyFont)) return 'sfx'

  // Signal B: extreme visual outlier → SFX in any script
  if (isVisualSfx(block, chars, blockFont, bodyFont)) return 'sfx'

  // Signal C: repeated short glyph run → SFX in any script
  if (isRepeatedGlyphSfx(block, chars)) return 'sfx'

  return 'dialogue'
}

export function blockSourceFontPx(block: TextBlock): number {
  const lineSamples = block.lines
    .map(line => block.textDirection === 'vertical' ? line.bbox[2] - line.bbox[0] : line.bbox[3] - line.bbox[1])
    .filter(n => n > 0)
  const wordSamples = block.words.map(word => Math.min(word.bbox[2] - word.bbox[0], word.bbox[3] - word.bbox[1])).filter(n => n > 0)
  const samples = (lineSamples.length ? lineSamples : wordSamples).sort((a, b) => a - b)
  return samples[Math.floor(samples.length / 2)] ?? 0
}

// ── Signal A: Katakana SFX ──────────────────────────────────────────────────

function isKatakanaSfx(block: TextBlock, chars: number, blockFont: number, bodyFont: number): boolean {
  if (chars > KATAKANA_MAX_CHARS || blockFont <= 0 || bodyFont <= 0) return false
  if (!isKatakanaDominant(block.text)) return false
  if (chars > 1 && !hasDecorativeAspect(block, KATAKANA_ASPECT)) return false
  if (!hasVisualScale(block, chars, blockFont, bodyFont, KATAKANA_FONT_RATIO)) return false
  return true
}

function isKatakanaDominant(text: string): boolean {
  let kata = 0
  let total = 0
  for (const c of text) {
    const cp = c.codePointAt(0)!
    if (cp <= 0x20 || (cp >= 0x3000 && cp <= 0x303F)) continue // space + CJK punctuation
    total++
    // Katakana block + Phonetic Extensions + Halfwidth
    if ((cp >= 0x30A0 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF) || (cp >= 0xFF65 && cp <= 0xFF9F)) kata++
  }
  return total >= 2 && kata / total >= 0.50
}

// ── Signal B: Visual SFX ────────────────────────────────────────────────────

function isVisualSfx(block: TextBlock, chars: number, blockFont: number, bodyFont: number): boolean {
  if (chars > VISUAL_MAX_CHARS || blockFont <= 0 || bodyFont <= 0) return false
  if (blockFont >= bodyFont * VISUAL_EXTREME_FONT_RATIO) return true
  if (chars > 1 && !hasDecorativeAspect(block, VISUAL_ASPECT)) return false
  if (!hasVisualScale(block, chars, blockFont, bodyFont, VISUAL_FONT_RATIO)) return false
  return true
}

// ── Signal C: Repeated glyph SFX ─────────────────────────────────────────────

function isRepeatedGlyphSfx(block: TextBlock, chars: number): boolean {
  if (chars < REPEATED_GLYPH_MIN_CHARS || chars > REPEATED_GLYPH_MAX_CHARS) return false
  const glyphs = [...block.text].filter(ch => !/[\s\p{P}\p{S}]/u.test(ch))
  if (glyphs.length !== chars) return false
  if (new Set(glyphs).size !== 1) return false
  return block.lines.length >= REPEATED_GLYPH_MIN_CHARS || hasDecorativeAspect(block, 1.15)
}

// ── Shared ──────────────────────────────────────────────────────────────────

function hasDecorativeAspect(block: TextBlock, minAspect: number): boolean {
  const w = Math.max(1, block.bbox[2] - block.bbox[0])
  const h = Math.max(1, block.bbox[3] - block.bbox[1])
  return Math.max(w / h, h / w) >= minAspect
}

function hasVisualScale(block: TextBlock, chars: number, blockFont: number, bodyFont: number, ratio: number): boolean {
  if (blockFont >= bodyFont * ratio) return true
  const w = Math.max(1, block.bbox[2] - block.bbox[0])
  const h = Math.max(1, block.bbox[3] - block.bbox[1])
  const longSide = Math.max(w, h)
  const expectedLongSide = bodyFont * Math.max(1, chars)
  return longSide >= expectedLongSide * ratio
}
