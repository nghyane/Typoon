// Work-centric chapter merge.
//
// One Work groups every sibling material. After community-vote
// merges, a single Work may carry N source-installed materials
// (HappyMH-zh + OTruyen-vi + MangaDex multilang). The chapter list is
// the UNION of every installed source's manifest spine, overlaid
// with cross-source translations the community has spawned.
//
// Two inputs:
//   • workDetail     — server payload (sibling materials, viewer
//                       entry, work_chapters with shared translations
//                       attached).
//   • manifestSources — every installed-source material's manifest
//                       chapter list. Caller fetches in parallel; a
//                       failed manifest fetch silently contributes
//                       no raws.
//
// Two kinds of versions per chapter:
//
//   • raw          a manifest-live chapter URL on ANY installed
//                   source. N source materials → up to N raws per
//                   chapter row.
//   • translation  a Work-chapter translation row from the server
//                   (cross-source, any sibling material's spawn).
//
// Merge key is `numberNorm` — the manifest runtime computes it once
// per chapter via the declarative normaliser. Server work_chapters
// carry the same key.
//
// Sort: chapter number desc (latest first). Caller may re-sort.
//
// Pure function — no React, no fetches. The hook layer drives all
// IO; this module just folds.

import type {
  ApiMaterial, ApiWorkDetail, DraftState,
} from '@shared/api/api'
import type {
  InstalledSource, MangaChapterRef,
} from '@features/browse/manifest/types'


export type VersionKind = 'raw' | 'translation'

export interface HubVersion {
  /** stable React key */
  key:            string
  kind:           VersionKind

  /** What lang the user reads in. raw = source lang; translation =
   *  target_lang. */
  lang:           string

  /** raw — the active source that owns the manifest URL.
   *  translation — the source whose pixels back the rendered draft
   *  (= the material the reader opens on click). */
  materialId:     number | null
  sourceId:       string | null
  sourceName:     string | null

  /** raw: manifest chapter URL the reader opens directly.
   *  translation: null. */
  upstreamUrl:    string | null

  /** raw: manifest-normalised chapter key (mirror of HubChapter.number).
   *  translation: null. */
  numberNorm:     string | null

  /** translation only — id + state + creator. */
  translationId:  number | null
  state:          DraftState | null
  /** Cause of the last failure (translation only). Worker stamps
   *  this on the draft when `state ∈ {'error','blocked'}` so the row
   *  can render the reason inline instead of hiding it behind a
   *  hover-only tooltip — mobile users would otherwise see nothing
   *  but a vague "Lỗi" chip. Null while pending/running/done. */
  errorMessage:   string | null
  /** Creator attribution. For typoon translations: `@user` display
   *  name. For raw rows: scanlator group / publisher (when the
   *  source exposes it via `ChapterFields.scanlator`). Renders the
   *  same way in both: a single "@name · MangaDex" badge. */
  creatorName:    string | null
  /** translation only — BCP-47 of the raw the draft was rendered
   *  from. Lets the row render "@userA · từ Tiếng Anh MangaDex". */
  sourceLang:     string | null
  /** ISO timestamp the underlying chapter was published (raw) or
   *  the translation was last updated. Used for the "3 ngày trước"
   *  badge on the row + the chapter-level `updatedAt` aggregate. */
  date:           string | null
  /** Whether the translation reuses the shared draft's render. */
  fromCache:      boolean
  /** Whether the translation is publicly visible (false → owner only). */
  shared:         boolean
}

export interface HubChapter {
  /** Canonical key — `work_chapters.number_norm` joined with the
   *  manifest's `MangaChapterRef.numberNorm`. */
  number:      string
  /** First non-null label among versions (server work_chapter.label
   *  if present, else manifest label). */
  label:       string | null
  /** Numeric sort key (parseFloat of number, NaN → 0). */
  sortKey:     number
  /** Newest version timestamp across all sources/translations; used
   *  for date sort. Null when nothing carries a timestamp. */
  updatedAt:   string | null
  versions:    HubVersion[]
}


/** One installed-source material's manifest chapter list, paired
 *  with the source plugin handle that resolved its name. The Work
 *  page may attach several of these — one per installed source
 *  material — and `mergeChapters` unions all their raws into a
 *  single chapter list.
 */
export interface ManifestSource {
  material: ApiMaterial
  source:   InstalledSource
  chapters: MangaChapterRef[]
}


export interface MergeInput {
  /** Server work payload — sibling materials + cross-source overlay. */
  work:             ApiWorkDetail
  /** Every installed-source material whose manifest has resolved.
   *  A material with a failed / in-flight manifest fetch is omitted
   *  by the caller (silent skip). */
  manifestSources:  ManifestSource[]
  /** Map of every installed source by manifest id. Used to resolve
   *  source names on translation versions whose draft was rendered
   *  from a sibling material (cross-source overlay). */
  installedSources: Record<string, InstalledSource>
}


export function mergeChapters(input: MergeInput): HubChapter[] {
  const { work, manifestSources, installedSources } = input

  const materialsById = new Map<number, ApiMaterial>()
  for (const m of work.materials) materialsById.set(m.id, m)

  const byNumber = new Map<string, HubChapter>()

  // 1) Manifest chapters of every installed source → raw versions.
  //    N source materials can each contribute a raw at the same
  //    numberNorm. The HubVersion key already embeds materialId so
  //    raws across sources stay unique.
  for (const ms of manifestSources) {
    const material = ms.material
    // Single-language sources (OTruyen, HappyMH) don't set per-chapter
    // language on `mc`; the source's static `manifest.languages[0]` is
    // the authoritative fallback. `material.languages` is a different
    // concept (what langs the source EXPOSES — used for browse filters
    // / badges) and is not a reliable signal for a chapter's lang.
    const sourceLang = ms.source.manifest.languages[0] ?? null
    for (const mc of ms.chapters) {
      const num = mc.numberNorm
      const ch  = ensureChapter(byNumber, num, mc.label)
      ch.versions.push({
        key:            `raw::${material.id}::${mc.id}`,
        kind:           'raw',
        lang:           normalizeLang(mc.language ?? sourceLang),
        materialId:     material.id,
        sourceId:       material.source,
        sourceName:     ms.source.manifest.name ?? material.source,
        upstreamUrl:    mc.url,
        numberNorm:     mc.numberNorm,
        translationId:  null,
        state:          null,
        errorMessage:   null,
        creatorName:    mc.scanlator,
        sourceLang:     null,
        date:           mc.date,
        fromCache:      false,
        shared:         false,
      })
      ch.updatedAt = newer(ch.updatedAt, mc.date)
    }
  }

  // 2) Work chapters from the server → translation versions, attached
  //    to the matching chapter row (creating it if no manifest row
  //    surfaced it — e.g. a chapter the active source dropped but
  //    a sibling still hosts).
  for (const wc of work.chapters) {
    const ch = ensureChapter(byNumber, wc.number_norm, wc.label)
    for (const t of wc.translations) {
      const draftMaterial = t.draft_material_id != null
        ? materialsById.get(t.draft_material_id) ?? null
        : null
      const draftSource = draftMaterial?.source
        ? (installedSources[draftMaterial.source] ?? null)
        : null
      ch.versions.push({
        key:            `tr::${t.id}`,
        kind:           'translation',
        lang:           normalizeLang(t.target_lang),
        materialId:     draftMaterial?.id ?? null,
        sourceId:       draftMaterial?.source ?? null,
        sourceName:     draftSource?.manifest.name ?? draftMaterial?.source ?? null,
        upstreamUrl:    null,
        numberNorm:     null,
        translationId:  t.id,
        state:          t.state,
        errorMessage:   t.error_message,
        creatorName:    t.creator_name,
        sourceLang:     t.source_lang,
        date:           t.updated_at,
        fromCache:      t.uses_default_render,
        shared:         t.shared,
      })
      ch.updatedAt = newer(ch.updatedAt, t.updated_at)
    }
  }

  const out = [...byNumber.values()]
  for (const ch of out) {
    // Deduplicate identical raw versions (defensive — manifest is the
    // only raw source so this should be a no-op, but keep the guard
    // in case future code paths layer raws again).
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


// ── Helpers ─────────────────────────────────────────────────────────


function parseSortKey(num: string): number {
  const v = parseFloat(num)
  return Number.isFinite(v) ? v : 0
}


// Lang code normalisation — accept "vi-VN", "VI", "vi" → "vi".
function normalizeLang(lang: string | null): string {
  if (!lang) return 'unknown'
  return lang.toLowerCase().split(/[-_]/)[0]
}


function newer(a: string | null, b: string | null | undefined): string | null {
  if (!b) return a
  if (!a) return b
  return a > b ? a : b
}


// ── Display helpers ────────────────────────────────────────────────


// ── Action helpers ───────────────────────────────────────────────


/** Per-chapter status across versions. Drives the "Đọc / Dịch /
 *  Đang dịch / Lỗi" affordance in the row.
 */
export type ChapterStatus = 'translated' | 'running' | 'error' | 'raw'
export type StatusFilter  = 'all' | ChapterStatus
export type ChapterSort   = 'chapter_desc' | 'chapter_asc' | 'updated_desc'


export function chapterStatus(
  ch: HubChapter,
  targetLang: string | null,
): ChapterStatus {
  if (!targetLang) return 'raw'
  if (preferredReadable(ch, targetLang)) return 'translated'
  if (inFlight(ch, targetLang))          return 'running'
  if (lastError(ch, targetLang))         return 'error'
  return 'raw'
}


export function preferredReadable(
  ch: HubChapter,
  targetLang: string | null,
): HubVersion | null {
  if (!targetLang) return null
  const lang = normalizeLang(targetLang)
  // 1) Translation done at target_lang (cross-source — any sibling).
  const tx = ch.versions.find(
    (v) => v.kind === 'translation' && v.lang === lang && v.state === 'done',
  )
  if (tx) return tx
  // 2) Raw at target_lang (native scanlation — no LLM needed).
  const raw = ch.versions.find((v) => v.kind === 'raw' && v.lang === lang)
  return raw ?? null
}


export function inFlight(
  ch: HubChapter,
  targetLang: string | null,
): HubVersion | null {
  if (!targetLang) return null
  const lang = normalizeLang(targetLang)
  return (
    ch.versions.find(
      (v) => v.kind === 'translation'
        && v.lang === lang
        && (v.state === 'pending' || v.state === 'running'),
    ) ?? null
  )
}


export function lastError(
  ch: HubChapter,
  targetLang: string | null,
): HubVersion | null {
  if (!targetLang) return null
  const lang = normalizeLang(targetLang)
  return (
    ch.versions.find(
      (v) => v.kind === 'translation'
        && v.lang === lang
        // `blocked` rolls into the user-facing error bucket: the
        // chapter can't progress until an admin clears `stage_pause`,
        // which from a reader's POV is indistinguishable from a hard
        // failure. We surface a distinct chip in the row itself; at
        // the chapter-status level both bubble up the same way.
        && (v.state === 'error' || v.state === 'blocked'),
    ) ?? null
  )
}


export function countByStatus(
  chapters:   HubChapter[],
  targetLang: string | null,
): Record<StatusFilter, number> {
  const out: Record<StatusFilter, number> = {
    all: chapters.length, translated: 0, running: 0, error: 0, raw: 0,
  }
  for (const c of chapters) out[chapterStatus(c, targetLang)]++
  return out
}


export function compareChapters(
  s: ChapterSort,
): (a: HubChapter, b: HubChapter) => number {
  switch (s) {
    case 'chapter_desc': return (a, b) => b.sortKey - a.sortKey
    case 'chapter_asc':  return (a, b) => a.sortKey - b.sortKey
    case 'updated_desc': return (a, b) =>
      (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
  }
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
