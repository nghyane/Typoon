import type { TextRole } from '../domain/planning'
import { classifyTextScript, type TextScript } from './textScript'

// Global size knob applied to every dialogue scale below (1 = as tuned). Raise
// for bolder/larger translations across the board, lower for smaller.
const VISUAL_GLYPH_MATCH = 1
// Dense sources (Hangul/CJK) pack a whole syllable/word into ONE square glyph,
// so the Latin (Vietnamese) translation of it is many more glyphs. Rendering at
// the full source glyph height then reads oversized and overflows the bubble —
// dense sources take a density scale-DOWN. Latin→Latin (EN→VI) is similar in
// length, so it stays near 1:1.
const DENSE_SOURCE_SCALE = 0.85
const LATIN_SOURCE_SCALE = 0.95

export interface RenderLanguageContext {
  readonly sourceLanguage?: string | null
  readonly targetLanguage?: string | null
}

export interface TextRenderProfile {
  readonly sourceFamily: LanguageFamily
  readonly targetFamily: LanguageFamily
  readonly targetScript: TextScript
  // T1 glyphScale: constant cross-script glyph-size ratio (source line box → VI
  // glyph). Proportional, so flat at every source size. "Already-small source"
  // is handled by minReadableFontPx (the floor), NOT by tapering this scale.
  readonly fontScale: number
  readonly minReadableFontPx: number
  // T1 leading: extra line-height multiplier so stacked diacritics (Vietnamese
  // tone + vowel marks) are not crowded/clipped between lines. 1.0 = none.
  readonly leadingRatio: number
  readonly innerPadXEm: number
  readonly innerPadYEm: number
  // T2 anchor-region growth: how much the source-text footprint is grown to make
  // room for the (usually longer) translation, before any bubble expansion.
  // Latin/VI expands more than CJK targets.
  readonly expansionAllowanceX: number
  readonly expansionAllowanceY: number
  readonly pageMaxFraction: number
  readonly hierarchyMaxFraction: number
}

export type LanguageFamily = 'latin' | 'hangul' | 'han' | 'kana' | 'mixed-cjk' | 'unknown'

export function textRenderProfile(
  text: string,
  context: RenderLanguageContext | undefined,
  role: TextRole,
  sourceText?: string | null,
): TextRenderProfile {
  const sourceScript = sourceText ? classifyTextScript(sourceText) : null
  const sourceFamily = languageFamily(context?.sourceLanguage, sourceScript)
  const targetScript = classifyTextScript(text)
  const targetFamily = languageFamily(context?.targetLanguage, targetScript)
  const latinTarget = targetFamily === 'latin'
  const syllabicTarget = targetFamily === 'hangul'
  const cjkTarget = targetFamily === 'han' || targetFamily === 'kana' || targetFamily === 'mixed-cjk'
  // glyphScale: target font-size = measured source glyph height × a density-aware
  // ratio. Dense sources (Hangul/CJK) render SMALLER (their one square glyph
  // becomes several Latin letters); Latin source stays near 1:1. The fit/expand
  // stage (T4) then grows the box just enough to hold this size, shrinking only
  // when the bubble truly can't.
  const denseSource = sourceFamily === 'hangul' || sourceFamily === 'han' || sourceFamily === 'kana' || sourceFamily === 'mixed-cjk'
  const fontScale = role === 'sfx' ? 1
    : latinTarget && denseSource ? DENSE_SOURCE_SCALE * VISUAL_GLYPH_MATCH
    : latinTarget ? LATIN_SOURCE_SCALE * VISUAL_GLYPH_MATCH
    : syllabicTarget ? 0.96
    : cjkTarget ? 1.02
    : 0.94

  return {
    sourceFamily,
    targetFamily,
    targetScript,
    fontScale,
    minReadableFontPx: role === 'sfx' ? 6 : latinTarget ? 10 : 8,
    leadingRatio: latinTarget ? 1.06 : 1.0,
    innerPadXEm: role === 'sfx' ? 0.10 : latinTarget ? 0.40 : syllabicTarget ? 0.34 : 0.28,
    innerPadYEm: role === 'sfx' ? 0.10 : latinTarget ? 0.24 : syllabicTarget ? 0.22 : 0.20,
    expansionAllowanceX: latinTarget ? 1.28 : 1.20,
    expansionAllowanceY: latinTarget ? 1.22 : 1.16,
    pageMaxFraction: latinTarget ? 0.045 : syllabicTarget ? 0.048 : 0.052,
    hierarchyMaxFraction: latinTarget ? 0.12 : syllabicTarget ? 0.13 : 0.14,
  }
}

export function pageRenderProfile(context: RenderLanguageContext | undefined): Pick<TextRenderProfile, 'pageMaxFraction' | 'hierarchyMaxFraction'> {
  const targetFamily = languageFamily(context?.targetLanguage, null)
  if (targetFamily === 'han' || targetFamily === 'kana' || targetFamily === 'mixed-cjk') {
    return { pageMaxFraction: 0.052, hierarchyMaxFraction: 0.14 }
  }
  if (targetFamily === 'hangul') return { pageMaxFraction: 0.048, hierarchyMaxFraction: 0.13 }
  return { pageMaxFraction: 0.045, hierarchyMaxFraction: 0.12 }
}

function languageFamily(language: string | null | undefined, fallbackScript: TextScript | null): LanguageFamily {
  const lang = language?.toLowerCase().trim() ?? ''
  if (/^(vi|en|fr|es|pt|pt-br|id|de|it|tr|ms)\b/u.test(lang)) return 'latin'
  if (/^ko\b/u.test(lang)) return 'hangul'
  if (/^ja\b/u.test(lang)) return 'kana'
  if (/^(zh|cmn|yue)\b/u.test(lang)) return 'han'
  if (!fallbackScript) return 'unknown'
  if (fallbackScript === 'latin') return 'latin'
  if (fallbackScript === 'hangul') return 'hangul'
  if (fallbackScript === 'han') return 'han'
  if (fallbackScript === 'kana') return 'kana'
  if (fallbackScript === 'mixed-cjk') return 'mixed-cjk'
  return 'unknown'
}
