import type { TextRole } from '../domain/planning'
import { classifyTextScript, type TextScript } from './textScript'

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
  readonly innerPadXEm: number
  readonly innerPadYEm: number
  readonly geometryGrowXEm: number
  readonly geometryGrowYEm: number
  readonly geometryGrowWidthRatio: number
  readonly geometryGrowHeightRatio: number
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
  const denseSource = sourceFamily === 'han' || sourceFamily === 'kana' || sourceFamily === 'mixed-cjk' || sourceFamily === 'hangul'
  // Latin source (EN/FR/…) → Latin target (VI): sourceFontPx is the OCR
  // line-box height (ascender→descender). Latin manga fonts (light/italic
  // narration, low x-height) leave the line box much taller than their visible
  // glyphs, so rendering our bold high-x-height target 1:1 against that height
  // overshoots ~30%. Scale down to the visible glyph size. CJK/Hangul glyphs are
  // square — their line box already equals the glyph — so they need no such
  // correction (only the density allowance below).
  const latinSource = sourceFamily === 'latin'
  const hangulSource = sourceFamily === 'hangul'

  // glyphScale: flat cross-script ratio. Long translations keep size via the
  // fit/expand stage (T4), never via a per-size taper here.
  const fontScale = role === 'sfx' ? 1
    : latinTarget && hangulSource ? 0.7
    : latinTarget && denseSource ? 0.88
    : latinTarget && latinSource ? 0.8
    : latinTarget ? 0.94
    : syllabicTarget ? 0.96
    : cjkTarget ? 1.02
    : 0.94

  return {
    sourceFamily,
    targetFamily,
    targetScript,
    fontScale,
    minReadableFontPx: role === 'sfx' ? 6 : latinTarget ? 10 : 8,
    innerPadXEm: role === 'sfx' ? 0.10 : latinTarget ? 0.40 : syllabicTarget ? 0.34 : 0.28,
    innerPadYEm: role === 'sfx' ? 0.10 : latinTarget ? 0.24 : syllabicTarget ? 0.22 : 0.20,
    geometryGrowXEm: latinTarget && denseSource ? 3.8 : latinTarget ? 3.1 : syllabicTarget ? 2.7 : 2.3,
    geometryGrowYEm: latinTarget ? 2.0 : syllabicTarget ? 1.8 : 1.6,
    geometryGrowWidthRatio: latinTarget && denseSource ? 0.42 : latinTarget ? 0.36 : 0.30,
    geometryGrowHeightRatio: latinTarget ? 0.30 : 0.24,
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
