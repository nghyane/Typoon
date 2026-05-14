// MangaDex link plugin — REST search at `api.mangadex.org`.
//
// MangaDex is JSON:API-flavoured: `attributes` carries the manga
// fields, `relationships[]` carries cover_art / author / artist as
// typed edges. Title is an object keyed by BCP-47 (typically one
// entry, e.g. `{ "ja-ro": "..." }`); `altTitles` is an array of
// single-key objects, one per known translation.
//
// We pick:
//   • title  — first non-null value from `attributes.title`.
//   • titleEnglish — `altTitles[*]@en`, first hit.
//   • titleNative  — `attributes.title.ja` / `altTitles[*]@ja`, first hit.
//   • synonyms     — every `altTitles[*]` value flattened.
//   • cover        — `https://uploads.mangadex.org/covers/{id}/{file}.512.jpg`.
//   • startYear    — `attributes.year`.
//
// The namespace `mdex_uuid` is already recognized by
// `isAlreadyEnriched`, so successful enrich here lets the linker
// auto-merge MangaDex siblings on the next vote round.

import {
  plinkFetch, toStr, toStrArray,
  type LinkCandidate, type LinkPluginAdapter, type LinkQuery,
} from '../runtime'


const ENDPOINT = 'https://api.mangadex.org/manga'
const COVER_BASE = 'https://uploads.mangadex.org/covers'
const LIMIT = 5


interface MangaDexRow {
  id?:            string
  attributes?: {
    title?:     Record<string, string> | null
    altTitles?: Record<string, string>[] | null
    year?:      number | null
  } | null
  relationships?: Array<{
    type?:       string
    attributes?: { fileName?: string | null } | null
  }> | null
}


export const mangadexPlugin: LinkPluginAdapter = {
  id:          'mangadex',
  name:        'MangaDex',
  namespace:   'mdex_uuid',
  description: 'REST API at api.mangadex.org — no auth required, generous CORS.',

  async search(q: LinkQuery, signal): Promise<LinkCandidate[]> {
    // MangaDex's title search is romanization-tolerant — pinyin /
    // romaji / english all hit. Native CJK matches occasionally too,
    // but the romanized title is the safer default. Fall back to
    // native when that's the only thing the caller has.
    const search = q.title.trim() || q.titleNative?.trim() || ''
    if (!search) return []
    try {
      const url = `${ENDPOINT}?title=${encodeURIComponent(search)}`
        + `&limit=${LIMIT}&includes[]=cover_art`
      const res = await plinkFetch(
        url,
        {
          method:  'GET',
          headers: { 'Accept': 'application/json' },
        },
        signal,
      )
      if (!res.ok) return []
      const json = await res.json() as { data?: MangaDexRow[] }
      const rows = json?.data ?? []
      return rows
        .map(mapRow)
        .filter((c): c is LinkCandidate => c !== null)
    } catch {
      return []
    }
  },
}


function mapRow(row: MangaDexRow): LinkCandidate | null {
  const id = row.id
  if (!id) return null

  const attrs    = row.attributes ?? {}
  const title    = pickFirstValue(attrs.title)
  const altList  = attrs.altTitles ?? []
  const altEn    = pickByLang(altList, 'en')
  const altJa    = pickByLang(altList, 'ja')
  const nativeFromTitle = attrs.title?.['ja']
                       ?? attrs.title?.['ko']
                       ?? attrs.title?.['zh']
                       ?? attrs.title?.['zh-hk']
                       ?? null

  const coverRel = (row.relationships ?? []).find(
    (r) => r.type === 'cover_art' && r.attributes?.fileName,
  )
  const coverFile = coverRel?.attributes?.fileName ?? null
  const cover = coverFile
    ? `${COVER_BASE}/${id}/${coverFile}.512.jpg`
    : null

  const year = attrs.year ?? null

  // Synonyms = every altTitle value EXCEPT the ones we already
  // surfaced as titleEnglish / titleNative. Keeps the synonym set
  // useful for similarity scoring without duplicating identity
  // fields.
  const allAlts = altList.flatMap((entry) => Object.values(entry))
  const synonyms = toStrArray(allAlts).filter(
    (s) => s !== altEn && s !== altJa && s !== title,
  )

  // Lang-tagged display titles: walk altTitles (one BCP-47 → string
  // per entry) and keep the first hit per primary lang subtag. The
  // primary `attributes.title` map is included too because MangaDex
  // sometimes ships only the romanized version there. This is what
  // makes `materials.title_locale[viewer_lang]` populated; the
  // resolver then surfaces e.g. the `vi` title for Vietnamese
  // viewers instead of falling through to title_native.
  const titleLocale: Record<string, string> = {}
  const addLocale = (rawLang: string, raw: string | null | undefined) => {
    const lang = rawLang.trim().toLowerCase().split('-')[0]
    if (!lang) return
    const v = (raw ?? '').trim()
    if (!v) return
    if (titleLocale[lang]) return  // first hit wins
    titleLocale[lang] = v
  }
  for (const [lang, v] of Object.entries(attrs.title ?? {})) {
    addLocale(lang, v)
  }
  for (const entry of altList) {
    for (const [lang, v] of Object.entries(entry)) {
      addLocale(lang, v)
    }
  }

  return {
    plugin:       'mangadex',
    namespace:    'mdex_uuid',
    externalId:   id,
    title:        toStr(title),
    titleEnglish: toStr(altEn),
    titleNative:  toStr(nativeFromTitle ?? altJa),
    titleLocale,
    synonyms,
    cover,
    startYear:    typeof year === 'number' && year > 0 ? year : null,
  }
}


/** First non-null value from a BCP-47 keyed object. MangaDex's
 *  `attributes.title` typically has ONE entry — whichever romanization
 *  the upstream prefers (`ja-ro`, `en`, etc.) — so picking the first
 *  is the right semantic. */
function pickFirstValue(
  obj: Record<string, string> | null | undefined,
): string | null {
  if (!obj) return null
  for (const v of Object.values(obj)) {
    if (typeof v === 'string' && v.trim()) return v
  }
  return null
}


/** Find the first altTitles entry keyed by `lang` (case-insensitive
 *  match on the language part). MangaDex altTitles are an array of
 *  single-key objects (one lang per entry), so we scan linearly. */
function pickByLang(
  list: Record<string, string>[],
  lang: string,
): string | null {
  const want = lang.toLowerCase()
  for (const entry of list) {
    for (const [k, v] of Object.entries(entry)) {
      if (k.toLowerCase().split('-')[0] === want
          && typeof v === 'string' && v.trim()) {
        return v
      }
    }
  }
  return null
}
