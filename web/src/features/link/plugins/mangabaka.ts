// MangaBaka link plugin — aggregator that resolves a single search
// query into cross-references for Anilist, MyAnimeList, MangaUpdates,
// Kitsu, AnimePlanet, AnimeNewsNetwork, Shikimori in ONE call. No
// auth, CORS-open at `api.mangabaka.dev`.
//
// Why this exists alongside `mangadexPlugin`: MangaDex maintains its
// own UUID namespace (`mdex_uuid`) that we wire directly through the
// manga-source manifest; MangaBaka covers everyone else. The two
// adapters together give the auto-enrich flow the full picture
// without standalone Anilist / MAL / MU adapters (Anilist's API is
// flaky, MAL has no public CORS, MU is XML).
//
// Identity model: each MangaBaka row carries an `id` (its own
// primary key) plus a `source` map where each entry's `.id` is the
// external service's identifier. We commit:
//
//   • mangabaka      — the row id itself
//   • anilist        — `source.anilist.id`
//   • mal            — `source.my_anime_list.id`
//   • mu             — `source.manga_updates.id`   (string slug)
//   • kitsu          — `source.kitsu.id`
//
// `isAlreadyEnriched` checks for `anilist | mdex_uuid | mal`, so as
// long as the upstream has at least one of those three, a successful
// MangaBaka enrich satisfies the "this Work has cross_refs" gate.
//
// Note on `merged_with`: MangaBaka occasionally consolidates rows
// upstream. When a row carries `merged_with` we skip it — the caller
// should re-search to land on the canonical id.

import {
  plinkFetch, toStr, toStrArray,
  type LinkCandidate, type LinkPluginAdapter, type LinkQuery,
} from '../runtime'


const ENDPOINT = 'https://api.mangabaka.dev/v1/series/search'
const LIMIT = 5

// MangaBaka source-map keys we forward into `cross_refs`. Each entry
// names the upstream service and the cross_refs namespace we want
// to commit it under. Order = priority when surfacing the "best"
// external id (we still commit ALL of them; this is just for
// `LinkCandidate.externalId` which expects ONE string).
const FORWARDED_SOURCES: Array<{ key: string; namespace: string }> = [
  { key: 'anilist',        namespace: 'anilist' },
  { key: 'my_anime_list',  namespace: 'mal' },
  { key: 'manga_updates',  namespace: 'mu' },
  { key: 'kitsu',          namespace: 'kitsu' },
]


interface MangaBakaRow {
  id?:               number
  state?:            string
  merged_with?:      number | null
  title?:            string | null
  native_title?:     string | null
  romanized_title?:  string | null
  type?:             string | null
  year?:             number | null
  authors?:          string[] | null
  cover?:            {
    raw?: { url?: string | null } | null
  } | null
  secondary_titles?: Record<string, Array<{
    type?:  string
    title?: string
    note?:  string
  }>> | null
  source?: Record<string, { id?: string | number | null }> | null
}


export const mangabakaPlugin: LinkPluginAdapter = {
  id:          'mangabaka',
  name:        'MangaBaka',
  namespace:   'mangabaka',
  description: 'Cross-reference aggregator at api.mangabaka.dev — Anilist + MAL + MangaUpdates + Kitsu in one call.',

  async search(q: LinkQuery, signal): Promise<LinkCandidate[]> {
    // MangaBaka indexes by every title variant it knows, including
    // native CJK. Prefer the native query when present — it bypasses
    // romanization ambiguity entirely.
    const search = (q.titleNative?.trim() || q.title.trim())
    if (!search) return []
    try {
      const url = `${ENDPOINT}?q=${encodeURIComponent(search)}&limit=${LIMIT}`
      const res = await plinkFetch(
        url,
        {
          method:  'GET',
          headers: { 'Accept': 'application/json' },
        },
        signal,
      )
      if (!res.ok) return []
      const json = await res.json() as { data?: MangaBakaRow[] }
      const rows = json?.data ?? []
      const out: LinkCandidate[] = []
      for (const row of rows) {
        const mapped = mapRow(row)
        if (mapped) out.push(...mapped)
      }
      return out
    } catch {
      return []
    }
  },
}


/** Map ONE MangaBaka row into multiple LinkCandidates — one per
 *  forwarded external service — so downstream `buildPayload` can
 *  commit each external id under its own namespace. Every candidate
 *  shares the SAME title/native/cover metadata; only `namespace` and
 *  `externalId` differ.
 *
 *  Returns null when the row is unusable (merged / no external ids).
 */
function mapRow(row: MangaBakaRow): LinkCandidate[] | null {
  if (!row.id) return null
  if (row.merged_with != null) return null            // skip merged duplicates
  if (row.state && row.state !== 'active') return null

  const sources = row.source ?? {}

  const titleNative   = toStr(row.native_title)
  const titleRoman    = toStr(row.romanized_title)
  // `title` is MangaBaka's preferred display string — usually the
  // romanized one. Fall back to romanized/native if absent.
  const titleDisplay  = toStr(row.title) ?? titleRoman ?? titleNative
  const titleEnglish  = pickEnglish(row.secondary_titles)
  // Lang-tagged display titles: one preferred title per BCP-47 code
  // MangaBaka has data for. Goes straight into `materials.title_locale`
  // so the resolver picks the viewer-lang variant ahead of the
  // romanized display fallback. Includes English (so a one-plugin
  // enrich still seeds `title_locale['en']`).
  const titleLocale   = pickLocaleMap(row.secondary_titles)
  const synonyms      = collectSynonyms(row.secondary_titles, {
    skip: [
      titleDisplay, titleNative, titleRoman, titleEnglish,
      ...Object.values(titleLocale),
    ].filter((s): s is string => !!s),
  })
  const cover     = toStr(row.cover?.raw?.url)
  const startYear = typeof row.year === 'number' && row.year > 0
    ? row.year
    : null

  const candidates: LinkCandidate[] = []

  // Always commit the MangaBaka id itself so we can route back to
  // the aggregator later (admin tooling, suggestion expansion).
  candidates.push({
    plugin:       'mangabaka',
    namespace:    'mangabaka',
    externalId:   String(row.id),
    title:        titleDisplay,
    titleEnglish,
    titleNative,
    titleLocale,
    synonyms,
    cover,
    startYear,
  })

  // Plus one candidate per forwarded external service that the row
  // actually has an id for.
  for (const { key, namespace } of FORWARDED_SOURCES) {
    const rawId = sources[key]?.id
    if (rawId == null) continue
    const idStr = String(rawId).trim()
    if (!idStr) continue
    candidates.push({
      plugin:       'mangabaka',
      namespace,
      externalId:   idStr,
      title:        titleDisplay,
      titleEnglish,
      titleNative,
      titleLocale,
      synonyms,
      cover,
      startYear,
    })
  }

  return candidates
}


/** Build a BCP-47 → preferred-title map from MangaBaka's
 *  `secondary_titles`. For each language code MangaBaka reports,
 *  pick the entry tagged `official` first, then `alternative`,
 *  then the first row. Lang codes are normalised lowercase, primary
 *  subtag only (`en-us` → `en`); when multiple regional variants of
 *  the same language carry titles, the first official wins.
 *
 *  This is the structural fix for the "title_alt has VI but
 *  resolver can't see it" bug: previously every secondary title got
 *  flattened into `synonyms` (lang signal lost). Now every official
 *  title flows into `title_locale` and the resolver finds the
 *  viewer's preferred language directly. */
function pickLocaleMap(
  titles: MangaBakaRow['secondary_titles'],
): Record<string, string> {
  if (!titles) return {}
  const out: Record<string, string> = {}
  for (const [rawLang, entries] of Object.entries(titles)) {
    if (!entries || entries.length === 0) continue
    const lang = rawLang.trim().toLowerCase().split('-')[0]
    if (!lang) continue
    // Skip 'ja-ro' / similar romanizations — they're stylistically
    // titleEnglish-equivalent, not a real ja display title.
    if (rawLang.toLowerCase().endsWith('-ro')) continue
    if (out[lang]) continue   // first locale wins
    const official    = entries.find((e) => e.type === 'official')
    const alternative = entries.find((e) => e.type === 'alternative')
    const t = official?.title ?? alternative?.title ?? entries[0]?.title
    if (t && t.trim()) out[lang] = t.trim()
  }
  return out
}


/** Find the first English-language entry in MangaBaka's
 *  secondary_titles map. Prefer `type === 'official'` over
 *  alternatives. */
function pickEnglish(
  titles: MangaBakaRow['secondary_titles'],
): string | null {
  if (!titles) return null
  for (const lang of ['en', 'en-us', 'en-gb']) {
    const entries = titles[lang] ?? titles[lang.toLowerCase()]
    if (!entries) continue
    const official    = entries.find((e) => e.type === 'official')
    const alternative = entries.find((e) => e.type === 'alternative')
    const t = official?.title ?? alternative?.title ?? entries[0]?.title
    if (t && t.trim()) return t.trim()
  }
  return null
}


/** Flatten every secondary title into a `string[]`, skipping the
 *  values we already surface as titleDisplay / native / romanized /
 *  English. Useful for similarity scoring without duplicating the
 *  identity fields. */
function collectSynonyms(
  titles: MangaBakaRow['secondary_titles'],
  opts:   { skip: string[] },
): string[] {
  if (!titles) return []
  const skipSet = new Set(opts.skip.map((s) => s.trim().toLowerCase()))
  const all: string[] = []
  for (const entries of Object.values(titles)) {
    for (const e of entries) {
      if (e.title) all.push(e.title)
    }
  }
  return toStrArray(all).filter((s) => !skipSet.has(s.trim().toLowerCase()))
}
