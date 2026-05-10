// Lightweight language detection from a title string.
//
// Mirror of ext/core/lang/detect.ts — kept duplicated rather than
// shared via a workspace package because the rules are 30 lines and
// drift would be obvious. Keep them in sync if you change one.

const RE_HIRAGANA = /[\u3040-\u309F]/
const RE_KATAKANA = /[\u30A0-\u30FF\uFF66-\uFF9F]/
const RE_HANGUL   = /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F]/
const RE_CJK      = /[\u4E00-\u9FFF\u3400-\u4DBF]/
const RE_LATIN    = /[A-Za-z]/

export type DetectedLang = 'ja' | 'ko' | 'zh' | 'en'

export function detectLang(text: string): DetectedLang | null {
  if (!text) return null
  if (RE_HANGUL.test(text)) return 'ko'
  if (RE_HIRAGANA.test(text) || RE_KATAKANA.test(text)) return 'ja'
  if (RE_CJK.test(text)) return 'zh'
  if (RE_LATIN.test(text)) return 'en'
  return null
}
