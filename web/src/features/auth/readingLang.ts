// Reading-language resolver — pure utilities, no React, no fetch.
//
// Storage layout (production model):
//
//   • `users.preferred_target_lang`   viewer default — set on
//     onboarding, edited in Settings + the avatar dropdown.
//
//   • `library_entries.target_lang`   per-Work override — only
//     meaningful when the viewer wants to read THIS manga in a
//     different language than their default.
//
// Resolver precedence (`resolveReadingLang`):
//
//   1. library_entries.target_lang  (per-Work override)
//   2. users.preferred_target_lang  (viewer default)
//   3. FALLBACK_LANG                (hard-coded last resort)
//
// The reader pulls from the same chain; no URL `?lang=` exists.
// Sharing a chapter URL respects the recipient's own preference,
// the same way Netflix shares an episode without forcing subtitle
// track.


/** Languages exposed in pickers. Keep tight — every entry must
 *  match one of the source-side `manifest.languages` values the
 *  fanout search emits, otherwise the reader resolver fails to pick
 *  a matching version. Extend with care. */
export const LANG_OPTIONS: ReadonlyArray<{ code: string; label: string }> = [
  { code: 'vi', label: 'Tiếng Việt' },
  { code: 'en', label: 'English'    },
  { code: 'ja', label: '日本語'      },
  { code: 'ko', label: '한국어'      },
  { code: 'zh', label: '中文'        },
]

export const FALLBACK_LANG = 'vi'


/** Resolve the reading language for a viewer on a specific Work.
 *  `entryLang` is the per-Work override; `userLang` is the viewer
 *  default. Returns `FALLBACK_LANG` only when both upstream values
 *  are absent. */
export function resolveReadingLang(
  entryLang: string | null | undefined,
  userLang:  string | null | undefined,
): string {
  const e = (entryLang ?? '').trim().toLowerCase()
  if (e) return e
  const u = (userLang ?? '').trim().toLowerCase()
  if (u) return u
  return FALLBACK_LANG
}
