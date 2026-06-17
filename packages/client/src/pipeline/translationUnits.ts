import type { TextRole } from '../domain/planning'
import type { TextUnit } from '../domain/text'
import type { TranslationKind, TranslationUnit } from '../domain/translation'

export function translationUnitsFromTextUnits(units: readonly TextUnit[]): TranslationUnit[] {
  return units.map(unit => {
    const role = unit.roleHint ?? 'dialogue'
    return {
      id: unit.id,
      pageIndex: unit.pageIndex,
      blockIds: unit.blockIds,
      sourceText: unit.sourceText,
      kind: translationKind(unit.sourceText, role),
      role,
    }
  })
}

function translationKind(sourceText: string, role: TextRole): TranslationKind {
  if (!sourceText.trim()) return 'skip'
  if (isTextOnlyNoise(sourceText)) return 'skip'
  return role === 'sfx' ? 'sfx' : 'dialogue'
}

function isTextOnlyNoise(sourceText: string): boolean {
  const text = sourceText.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
  if (!text) return !isMeaningfulPunctuationOnly(sourceText)
  if (isStandaloneNonCjkNoise(text)) return true
  if (/https?|www|\.com|baozimh|包子漫[画畫]|騰訊動漫|腾讯动漫|tencent(?:anime|comics?)/iu.test(sourceText)) return true
  if (text.includes('最新免费漫画') || text.includes('最新免費漫畫')) return true
  return false
}

function isMeaningfulPunctuationOnly(sourceText: string): boolean {
  const compact = sourceText.replace(/\s+/gu, '')
  if (!compact || !/^[\p{P}\p{S}]+$/u.test(compact)) return false
  return /…|⋯|\.{2,}|。{2,}|[!?！？]{2,}|[—~〜]{2,}/u.test(compact)
}

function isStandaloneNonCjkNoise(text: string): boolean {
  return [...text].length === 1
    && !/[\p{Script=Latin}\p{N}\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(text)
}
