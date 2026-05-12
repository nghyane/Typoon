// Cross-material chapter merge.
//
// A library entry can link N materials (different sources, different
// raw languages). The hub renders ONE row per chapter number, with
// every available version stacked underneath as an action choice.
//
// Two kinds of versions per chapter:
//
//   • raw          a manifest-live chapter URL on some source.
//                   `lang` = the source language (per-chapter when the
//                   manifest exposes it, falling back to material
//                   languages[0]). When the user's target_lang has a
//                   raw version, they can read it directly — no LLM.
//   • translation  an LLM-driven translation row in DB (cross-user
//                   cached drafts surface as `from_cache=true`).
//
// Merge key is the normalised chapter number (strip leading zeros).
// Sort: chapter number desc (latest first) by default. Caller sorts
// downstream if a different order is needed.
//
// Pure function — no React, no fetches. The hook layer drives all
// IO; this module just folds.

import type {
  ApiMaterial, ApiMaterialDetail, DraftState,
} from '@shared/api/api'
import type {
  InstalledSource, MangaDetail,
} from '@features/browse/manifest/types'

/** Per-material input. `manifest` is null for ext / upload materials
 *  and for source-backed materials whose live fetch hasn't landed
 *  yet (or failed). */
export interface MaterialBundle {
  detail:    ApiMaterialDetail
  manifest:  MangaDetail | null
  source:    InstalledSource | null
}

export type VersionKind = 'raw' | 'translation'

export interface HubVersion {
  /** stable React key */
  key:            string
  kind:           VersionKind

  /** What lang the user reads in. raw = source lang; translation =
   *  target_lang. */
  lang:           string

  /** Which material this version comes from. */
  materialId:     number
  /** Source manifest id; null for ext / upload materials. */
  sourceId:       string | null
  /** Display label for the source ("HappyMH" / "MangaDex"); null
   *  for ext / upload. */
  sourceName:     string | null

  /** raw: manifest chapter URL the reader opens directly.
   *  translation: null. */
  upstreamUrl:    string | null

  /** translation only — id + state + creator. */
  translationId:  number | null
  state:          DraftState | null
  creatorName:    string | null
  /** Whether the translation reuses the shared draft's render. */
  fromCache:      boolean

  /** DB chapter id when materialized. Always present for translations;
   *  present for raws only when a user has touched the chapter (spawn
   *  / upload). */
  chapterId:      number | null
}

export interface HubChapter {
  /** Normalised number — strip leading zeros for grouping but keep
   *  decimals ('1.5', '12'). */
  number:      string
  /** First non-null label among versions. */
  label:       string | null
  /** Numeric sort key (parseFloat of number, NaN → 0). */
  sortKey:     number
  /** Newest version timestamp across all sources/translations; used
   *  for date sort. Null when nothing carries a timestamp. */
  updatedAt:   string | null
  versions:    HubVersion[]
}


export function mergeChapters(bundles: MaterialBundle[]): HubChapter[] {
  const byNumber = new Map<string, HubChapter>()

  for (const b of bundles) {
    const m = b.detail.material
    // upstream_url → DB chapter (for translation overlay).
    const dbByUrl = new Map<string, ApiMaterialDetail['chapters'][number]>()
    for (const c of b.detail.chapters) {
      if (c.upstream_url) dbByUrl.set(c.upstream_url, c)
    }

    // 1) Manifest chapters → raw versions + their translations.
    const seenDbIds = new Set<number>()
    if (b.manifest) {
      for (const mc of b.manifest.chapters) {
        const num = normalizeNumber(mc.number)
        const ch  = ensureChapter(byNumber, num, mc.label)

        const dbHit = dbByUrl.get(mc.url)
        if (dbHit) seenDbIds.add(dbHit.id)

        ch.versions.push({
          key:            `raw::${m.id}::${mc.id}`,
          kind:           'raw',
          lang:           normalizeLang(mc.language ?? m.languages[0]),
          materialId:     m.id,
          sourceId:       m.source,
          sourceName:     b.source?.manifest.name ?? m.source,
          upstreamUrl:    mc.url,
          translationId:  null,
          state:          null,
          creatorName:    null,
          fromCache:      false,
          chapterId:      dbHit?.id ?? null,
        })

        if (dbHit) {
          for (const t of dbHit.translations) {
            ch.versions.push(translationVersion(t, m, b.source, dbHit.id))
          }
          ch.updatedAt = newer(ch.updatedAt, dbHit.updated_at)
        }
      }
    }

    // 2) DB chapters NOT covered by the manifest (uploads, extras,
    //    or source returned a slimmer list). Treat them as raws too
    //    when an upstream_url exists, otherwise translation-only.
    for (const c of b.detail.chapters) {
      if (seenDbIds.has(c.id)) continue
      const num = normalizeNumber(c.number)
      const ch  = ensureChapter(byNumber, num, c.label)

      if (c.upstream_url) {
        ch.versions.push({
          key:            `raw::db::${c.id}`,
          kind:           'raw',
          lang:           normalizeLang(m.languages[0]),
          materialId:     m.id,
          sourceId:       m.source,
          sourceName:     b.source?.manifest.name ?? m.source,
          upstreamUrl:    c.upstream_url,
          translationId:  null,
          state:          null,
          creatorName:    null,
          fromCache:      false,
          chapterId:      c.id,
        })
      }
      for (const t of c.translations) {
        ch.versions.push(translationVersion(t, m, b.source, c.id))
      }
      ch.updatedAt = newer(ch.updatedAt, c.updated_at)
    }
  }

  const out = [...byNumber.values()]
  // Deduplicate identical raw versions (same materialId × upstreamUrl).
  for (const ch of out) {
    const seen = new Set<string>()
    ch.versions = ch.versions.filter((v) => {
      const k = v.kind === 'raw'
        ? `raw:${v.materialId}:${v.upstreamUrl}`
        : `tr:${v.translationId}`
      if (seen.has(k)) return false
      seen.add(k)
      return true
    })
  }

  return out.sort((a, b) => b.sortKey - a.sortKey)
}


function ensureChapter(
  byNumber: Map<string, HubChapter>,
  num:      string,
  label:    string | null,
): HubChapter {
  let ch = byNumber.get(num)
  if (!ch) {
    ch = {
      number:    num,
      label,
      sortKey:   parseSortKey(num),
      updatedAt: null,
      versions:  [],
    }
    byNumber.set(num, ch)
  } else if (!ch.label && label) {
    ch.label = label
  }
  return ch
}


function translationVersion(
  t: ApiMaterialDetail['chapters'][number]['translations'][number],
  material: ApiMaterial,
  source:   InstalledSource | null,
  chapterId: number,
): HubVersion {
  return {
    key:            `tr::${t.id}`,
    kind:           'translation',
    lang:           normalizeLang(t.target_lang),
    materialId:     material.id,
    sourceId:       material.source,
    sourceName:     source?.manifest.name ?? material.source,
    upstreamUrl:    null,
    translationId:  t.id,
    state:          t.state,
    creatorName:    t.creator_name,
    fromCache:      t.from_cache,
    chapterId,
  }
}


// ── Helpers ─────────────────────────────────────────────────────────

function normalizeNumber(n: string): string {
  const trimmed = n.trim()
  // Strip leading zeros from the integer part: '001' → '1', '01.5' → '1.5'.
  return trimmed.replace(/^0+(?=\d)/, '')
}

function parseSortKey(num: string): number {
  const v = parseFloat(num)
  return Number.isFinite(v) ? v : 0
}

function normalizeLang(lang: string | null | undefined): string {
  if (!lang) return '?'
  const lower = lang.trim().toLowerCase()
  // MangaDex emits region tags ('pt-br'); keep the base lang for
  // matching against target_lang ('vi'). Region info isn't surfaced
  // anywhere — drop it.
  return lower.split('-')[0] ?? lower
}

function newer(a: string | null, b: string | null | undefined): string | null {
  if (!b) return a
  if (!a) return b
  return a > b ? a : b
}


// ── Selectors over a HubChapter ─────────────────────────────────────

/** Translation versions only. */
export function chapterTranslations(ch: HubChapter): HubVersion[] {
  return ch.versions.filter((v) => v.kind === 'translation')
}

/** Distinct langs the user can read (any kind). */
export function chapterLangs(ch: HubChapter): string[] {
  const set = new Set<string>()
  for (const v of ch.versions) {
    if (v.kind === 'translation' && v.state !== 'done') continue
    set.add(v.lang)
  }
  return [...set]
}

/** Pick the version a user with this target_lang should land on by
 *  default. Priority:
 *    1. Translation done at target_lang (latest = first since the
 *       manifest order puts newer first).
 *    2. Raw at target_lang (native scanlation — no LLM needed).
 *    3. null when neither — caller renders the Dịch CTA instead. */
export function preferredReadable(
  ch: HubChapter,
  targetLang: string | null,
): HubVersion | null {
  if (!targetLang) return null
  const tgt = normalizeLang(targetLang)
  const tx = ch.versions.find(
    (v) => v.kind === 'translation' && v.state === 'done' && v.lang === tgt,
  )
  if (tx) return tx
  const raw = ch.versions.find((v) => v.kind === 'raw' && v.lang === tgt)
  return raw ?? null
}

/** Whether a translation is currently running for the user's target. */
export function inFlight(ch: HubChapter, targetLang: string | null): HubVersion | null {
  if (!targetLang) return null
  const tgt = normalizeLang(targetLang)
  return ch.versions.find(
    (v) => v.kind === 'translation'
        && v.lang === tgt
        && (v.state === 'running' || v.state === 'pending'),
  ) ?? null
}

/** Most-recent error for the user's target. */
export function lastError(ch: HubChapter, targetLang: string | null): HubVersion | null {
  if (!targetLang) return null
  const tgt = normalizeLang(targetLang)
  return ch.versions.find(
    (v) => v.kind === 'translation' && v.lang === tgt && v.state === 'error',
  ) ?? null
}
