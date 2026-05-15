// Chapter-row derivation — pure selectors over the merged Hub model.
//
// One Work → many chapters; each chapter renders as ONE row in the
// chapter list. The row carries everything the UI needs to render a
// button + meta line:
//
//   • status     — discriminated state for the action button
//   • read       — version to open when the button is "Đọc"
//   • spawnFrom  — raw version to upload when the button is "Dịch"
//   • translation — the most relevant translation row at target_lang
//                   (latest by score), used to surface server-side
//                   state (pending/running/error/blocked)
//
// "Most relevant" raw / translation picks:
//
//   read           prefer translation `done` → raw target-lang
//   spawnFrom      first raw at target-lang scan; else best non-target
//                  raw with an upstream URL (picks lang by score so
//                  EN > KR > others — closer to source-lang of most
//                  manga); skip raws whose `(source_lang, material)`
//                  already has a translation at target-lang
//   translation    prefer `done` → `running` → `pending` → `error` →
//                  `blocked` at target_lang
//
// Pure + memoizable: callers pass a stable `chapters` array and a
// stable `targetLang` and the row list stays referentially stable as
// long as inputs don't change.

import type { HubChapter, HubVersion } from './mergeChapters'


/** Discriminated state for the chapter row's primary affordance.
 *  Drives the button label, icon, color, and click handler.
 *
 *  States are mutually exclusive — `deriveChapterRow` picks exactly
 *  one based on the chapter's versions and the viewer's target_lang. */
export type ChapterRowStatus =
  | { kind: 'read-translation';  via: HubVersion }
  | { kind: 'read-raw-target';   via: HubVersion }
  | { kind: 'translatable';      from: HubVersion }
  | { kind: 'translating-server'; via: HubVersion }
  | { kind: 'translation-error';  via: HubVersion }
  | { kind: 'translation-blocked'; via: HubVersion }
  | { kind: 'unavailable' }


export interface ChapterRow {
  chapter:   HubChapter
  status:    ChapterRowStatus
  /** Best version to open on a "read" click. Null when `unavailable`. */
  read:      HubVersion | null
  /** Best raw to upload on a "spawn" click. Null when not translatable. */
  spawnFrom: HubVersion | null
  /** Translation overlay for this chapter at target_lang, or null when
   *  none exists. Used by the UI to surface creator + server state
   *  even when the primary button is "read-raw-target". */
  translation: HubVersion | null
  /** Versions surfaced in the "nguồn khác" pivot — every raw + every
   *  translation, in stable display order. Empty for `unavailable`. */
  alternates: HubVersion[]
}


export interface DeriveOpts {
  /** Set of `sourceId` strings whose plugin is installed. A raw whose
   *  source isn't installed can't be read (no manifest endpoints) and
   *  can't be spawned (no page list), so it's excluded from picks. */
  installedSourceIds: Set<string>
}


export function deriveChapterRow(
  chapter:    HubChapter,
  targetLang: string,
  opts:       DeriveOpts,
): ChapterRow {
  const lang = normalizeLang(targetLang)
  const versions = chapter.versions

  const readableRaw  = (v: HubVersion) =>
    v.kind === 'raw'
    && !!v.upstreamUrl
    && !!v.sourceId
    && opts.installedSourceIds.has(v.sourceId)

  // Translations at target_lang, ordered by interest.
  const translations = versions.filter(
    (v) => v.kind === 'translation' && v.lang === lang,
  )
  // Sort: done first, then running/pending/error/blocked; within the
  // same state newer first.
  const trScore = (v: HubVersion): number => {
    if (v.kind !== 'translation') return 99
    switch (v.state) {
      case 'done':    return 0
      case 'running': return 1
      case 'pending': return 2
      case 'error':   return 3
      case 'blocked': return 4
      default:        return 5
    }
  }
  translations.sort((a, b) => {
    const d = trScore(a) - trScore(b)
    if (d !== 0) return d
    return (b.date ?? '').localeCompare(a.date ?? '')
  })
  const translation = translations[0] ?? null

  // Raw at target lang — native scanlation reads without LLM.
  const rawTarget = versions.find(
    (v) => readableRaw(v) && v.lang === lang,
  ) ?? null

  // Raws at non-target langs, ranked by translation suitability.
  // We block raws whose `(source_lang, material_id)` already has a
  // translation at target_lang — re-spawning that exact draft would
  // just hit the server's UNIQUE cache and look like a no-op.
  const blocked = new Set<string>()
  for (const v of translations) {
    if (v.materialId != null && v.sourceLang) {
      blocked.add(`${v.sourceLang}::${v.materialId}`)
    }
  }
  const langRank = (l: string): number => {
    // EN > KR > JA > ZH > others. Closer to the source language of
    // most manga so the LLM has the highest-quality input.
    if (l === 'en') return 0
    if (l === 'ko') return 1
    if (l === 'ja') return 2
    if (l === 'zh') return 3
    return 10
  }
  const spawnFrom = versions
    .filter((v) =>
      readableRaw(v)
      && v.lang !== lang
      && !blocked.has(`${v.lang}::${v.materialId}`)
    )
    .sort((a, b) => langRank(a.lang) - langRank(b.lang))[0] ?? null

  // Status — first match wins. The list mirrors the user-facing
  // priority: a translated chapter reads as "done" even if a sibling
  // material is mid-upload.
  let status: ChapterRowStatus
  let read: HubVersion | null = null

  const tDone    = translations.find((v) => v.state === 'done')
  const tRunning = translations.find((v) => v.state === 'pending' || v.state === 'running')
  const tError   = translations.find((v) => v.state === 'error')
  const tBlocked = translations.find((v) => v.state === 'blocked')

  if (tDone) {
    status = { kind: 'read-translation', via: tDone }
    read = tDone
  } else if (rawTarget) {
    status = { kind: 'read-raw-target', via: rawTarget }
    read = rawTarget
  } else if (tRunning) {
    status = { kind: 'translating-server', via: tRunning }
    // Reader fallback: raw matching the running translation's source
    // lang so user can read what's being translated while waiting.
    read = pickRawByLang(versions, tRunning.sourceLang, opts.installedSourceIds)
        ?? pickAnyReadable(versions, opts.installedSourceIds)
  } else if (tError) {
    status = { kind: 'translation-error', via: tError }
    read = pickRawByLang(versions, tError.sourceLang, opts.installedSourceIds)
        ?? pickAnyReadable(versions, opts.installedSourceIds)
  } else if (tBlocked) {
    status = { kind: 'translation-blocked', via: tBlocked }
    read = pickRawByLang(versions, tBlocked.sourceLang, opts.installedSourceIds)
        ?? pickAnyReadable(versions, opts.installedSourceIds)
  } else if (spawnFrom) {
    status = { kind: 'translatable', from: spawnFrom }
    read = spawnFrom
  } else {
    status = { kind: 'unavailable' }
  }

  // Alternates — every readable version except the primary one. Used
  // by the row's "nguồn khác" pivot (dropdown / expanded section).
  // Stable ordering: translations done first, then raws by langRank.
  const alternates: HubVersion[] = []
  const primary = status.kind === 'read-translation' ? status.via
                : status.kind === 'read-raw-target'   ? status.via
                : status.kind === 'translatable'      ? status.from
                : null
  for (const v of translations) {
    if (v === primary) continue
    alternates.push(v)
  }
  for (const v of versions) {
    if (v === primary) continue
    if (!readableRaw(v)) continue
    if (alternates.includes(v)) continue
    alternates.push(v)
  }

  return { chapter, status, read, spawnFrom, translation, alternates }
}


export function deriveChapterRows(
  chapters:   HubChapter[],
  targetLang: string,
  opts:       DeriveOpts,
): ChapterRow[] {
  const out: ChapterRow[] = new Array(chapters.length)
  for (let i = 0; i < chapters.length; i++) {
    out[i] = deriveChapterRow(chapters[i]!, targetLang, opts)
  }
  return out
}


// ── Helpers ────────────────────────────────────────────────────


function pickRawByLang(
  versions: HubVersion[],
  lang:     string | null,
  installed: Set<string>,
): HubVersion | null {
  if (!lang) return null
  for (const v of versions) {
    if (v.kind !== 'raw') continue
    if (v.lang !== lang) continue
    if (!v.upstreamUrl || !v.sourceId) continue
    if (!installed.has(v.sourceId)) continue
    return v
  }
  return null
}


function pickAnyReadable(
  versions:  HubVersion[],
  installed: Set<string>,
): HubVersion | null {
  for (const v of versions) {
    if (v.kind !== 'raw') continue
    if (!v.upstreamUrl || !v.sourceId) continue
    if (!installed.has(v.sourceId)) continue
    return v
  }
  return null
}


function normalizeLang(lang: string): string {
  return lang.toLowerCase().split(/[-_]/)[0] ?? ''
}
