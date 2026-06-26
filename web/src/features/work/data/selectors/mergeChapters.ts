// Merged chapter list — cross-source dedupe by `numberNorm`.
//
// Pure: takes raw source chapters and returns deduplicated rows with
// all versions attached. Uses Map for O(1) lookup during merge, then
// materializes to array.

import type {
  MergedChapter, SourceChapterDetail, SourceVersion,
} from '../types'


export function mergeChapters(
  sourceChapters: SourceChapterDetail[],
  targetLang:     string,
): MergedChapter[] {
  const map = new Map<string, MergedChapter>()

  for (const sc of sourceChapters) {
    for (const ref of sc.refs) {
      const lang = (ref.language ?? sc.origin.languages[0] ?? targetLang).toLowerCase()
      const version: SourceVersion = {
        source: sc.source, origin: sc.origin, ref, lang,
      }
      const cur = map.get(ref.numberNorm)
      if (cur) {
        cur.sourceVersions.push(version)
      } else {
        map.set(ref.numberNorm, {
          numberNorm:     ref.numberNorm,
          label:          ref.label,
          number:         ref.number || ref.numberNorm,
          sortKey:        parseSortKey(ref.numberNorm),
          sourceVersions: [version],
        })
      }
    }
  }

  return [...map.values()]
}


/** Pick the best source version for a chapter when filtering by
 *  target lang. Prefers (a) target lang match, (b) newest date. */
export function pickBestVersion(
  ch: MergedChapter,
  targetLang: string,
): SourceVersion | null {
  if (ch.sourceVersions.length === 0) return null
  const target = ch.sourceVersions.filter(v => v.lang === targetLang)
  const pool   = target.length > 0 ? target : ch.sourceVersions
  return [...pool].sort(byDateDesc)[0] ?? null
}


function byDateDesc(a: SourceVersion, b: SourceVersion): number {
  const ad = a.ref.date ?? ''
  const bd = b.ref.date ?? ''
  return bd.localeCompare(ad)
}


function parseSortKey(numberNorm: string): number {
  const n = parseFloat(numberNorm)
  return Number.isFinite(n) ? n : 0
}
