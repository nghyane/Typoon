// Human-readable label for a BCP-47 language code. Used in language
// pickers, Tag chips, source descriptions. Falls back to upper-cased
// code when unknown so we never render a blank label.

const LANG_LABEL: Record<string, string> = {
  en:    'English',
  vi:    'Tiếng Việt',
  ja:    'Tiếng Nhật',
  ko:    'Tiếng Hàn',
  zh:    'Tiếng Trung',
  'zh-hk': '中文 (HK)',
  'pt-br': 'Português (BR)',
  pt:    'Português',
  es:    'Español',
  'es-la': 'Español (LA)',
  fr:    'Français',
  de:    'Deutsch',
  it:    'Italiano',
  ru:    'Русский',
  id:    'Indonesia',
  th:    'ภาษาไทย',
  ar:    'العربية',
  tr:    'Türkçe',
  pl:    'Polski',
  nl:    'Nederlands',
  fa:    'فارسی',
  hi:    'हिन्दी',
  bn:    'বাংলা',
  he:    'עברית',
  hu:    'Magyar',
  ms:    'Bahasa Melayu',
  tl:    'Filipino',
  el:    'Ελληνικά',
  ro:    'Română',
  uk:    'Українська',
  cs:    'Čeština',
  sv:    'Svenska',
  kk:    'Қазақша',
  la:    'Latīna',
  ka:    'ქართული',
}

export function languageName(code: string): string {
  return LANG_LABEL[code] ?? LANG_LABEL[code.toLowerCase()] ?? code.toUpperCase()
}

/** Short uppercased code for compact tags ("EN", "ZH-HK"). */
export function languageCode(code: string): string {
  return code.toUpperCase()
}

/** Multi-language sentinel used by manifests where the source spans
 *  many languages (MangaDex). Treated specially in UI. */
export const MULTI_LANG = 'multi'

export function languageSummary(codes: readonly string[]): string {
  if (codes.length === 0) return ''
  if (codes.length === 1) return languageName(codes[0]!)
  if (codes.length <= 3) return codes.map(languageCode).join(' · ')
  return `${codes.length} ngôn ngữ`
}
