// Lightweight language detection from a title string.
//
// Scope: meta-title hints from manga reader pages — short strings
// (often <100 chars) where Unicode block coverage is a near-perfect
// signal. We don't need a full classifier (CLD3, fasttext) for this:
//
//   ja → has hiragana or katakana   (kanji alone is ambiguous → zh)
//   ko → has hangul
//   zh → has CJK ideograph + no ja/ko marker
//   en → has Latin letters and no CJK at all
//
// Returns the ISO 639-1 code our project schema expects, or null
// when the text has no usable signal (empty / numbers only / mixed
// noise). Caller decides what to do with null — typically fall
// back to the previously chosen value.

const RE_HIRAGANA = /[\u3040-\u309F]/
const RE_KATAKANA = /[\u30A0-\u30FF\uFF66-\uFF9F]/
const RE_HANGUL   = /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F]/
const RE_CJK      = /[\u4E00-\u9FFF\u3400-\u4DBF]/   // ideographs (han)
const RE_LATIN    = /[A-Za-z]/

export type DetectedLang = 'ja' | 'ko' | 'zh' | 'en'

export function detectLang(text: string): DetectedLang | null {
  if (!text) return null

  // Order matters: a Japanese page with kanji + kana must not
  // collapse to zh. Korean is unambiguous (hangul is unique).
  if (RE_HANGUL.test(text)) return 'ko'
  if (RE_HIRAGANA.test(text) || RE_KATAKANA.test(text)) return 'ja'
  if (RE_CJK.test(text)) return 'zh'
  if (RE_LATIN.test(text)) return 'en'
  return null
}
