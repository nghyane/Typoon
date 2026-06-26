// Primary source resolver — picks which attached source's manifest
// drives the Work hub's identity meta (cover, author, description,
// status), and also seeds the auto-title that fills `work.title`
// until the user renames it explicitly.
//
// Why a Latin-aware fallback: a VN viewer attaching `JP raw` first
// would otherwise see "転スラ" as author/description language until
// they happen to attach a VI source. Promoting Latin script (en, vi)
// over CJK script keeps the meta readable for the majority case
// without forcing the user to reorder anything.
//
// Priority (highest score wins; ties → earliest in `work.sources[]`):
//
//   user-lang exact match           +100
//   Latin family   (user is Latin)  +20
//   has cover_url                   +5
//   index in sources                -i      (stable order tiebreak)
//
// `user-lang` is `work.target_lang` (per-Work override) or the
// viewer's `preferred_target_lang` — already collapsed by the
// caller. Latin family = en/vi/id/ms/fr/de/es/pt — anything written
// in the Latin script the viewer can sound out.

import type { WorkSource } from '@shared/db'


const LATIN_LANGS = new Set([
  'vi', 'en', 'id', 'ms', 'tl', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'pl',
  'tr', 'ro', 'cs', 'sv', 'no', 'fi', 'da', 'hu',
])


/** Returns the index of the primary source within `sources`. Returns
 *  `-1` for an empty array — callers handle that as "no sources". */
export function pickPrimarySourceIndex(
  sources:  WorkSource[],
  userLang: string,
): number {
  if (sources.length === 0) return -1
  const ul   = userLang.toLowerCase()
  const userIsLatin = LATIN_LANGS.has(ul)

  let bestIdx   = 0
  let bestScore = -Infinity

  for (let i = 0; i < sources.length; i++) {
    const s = sources[i]!
    const langs = s.languages.map(l => l.toLowerCase())
    let score = -i
    if (langs.includes(ul))                                 score += 100
    if (userIsLatin && langs.some(l => LATIN_LANGS.has(l))) score += 20
    if (s.cover_url)                                        score += 5

    if (score > bestScore) {
      bestScore = score
      bestIdx   = i
    }
  }

  return bestIdx
}


/** Pick the title to seed `work.title` from. Returns the primary
 *  source's title, or empty string if there are no sources.
 *
 *  Server-rendered library/recent views need a title without loading
 *  manifest details, so this seeds the local cache. Once the user
 *  renames explicitly (sets `title_overridden`), this is no longer
 *  consulted on attach/detach. */
export function pickAutoTitle(
  sources:  WorkSource[],
  userLang: string,
): string {
  const idx = pickPrimarySourceIndex(sources, userLang)
  return idx >= 0 ? sources[idx]!.title.trim() : ''
}

