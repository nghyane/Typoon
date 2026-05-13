// Work title — resolve the canonical display string for a Work from
// its sibling materials.
//
// Works don't carry their own title row (per the "Cách 1 — danh bạ"
// decision: Work = identity only, materials = display). The picker
// runs every viewer through the same priority chain so the title in
// the URL bar matches what got shared:
//
//   1. Material whose `languages` contains 'en'              ← universal
//   2. Material whose `title_native` is populated (romanized
//      title is the most reliable cross-language anchor we have
//      on a non-English material)                            ← fallback
//   3. `materials[0]` (oldest by id)                         ← default
//
// `title_native` for the subtitle is the FIRST material that carries
// one — Japanese / Korean / Chinese titles are the strongest
// cross-source identity hint.
//
// We deliberately don't let viewer's `target_lang` reorder the
// priority: showing a different title per viewer hides the canonical
// identity and breaks the "share this link, friend sees same thing"
// expectation. The user's reading language belongs in the chapter
// list, not in the page header.

import type { ApiMaterial } from '@shared/api/api'


export interface ResolvedWorkTitle {
  title:       string
  titleNative: string | null
  /** Which material the chosen title came from — exposed so the UI
   *  can show "title from MangaDex" if it ever wants attribution. */
  sourceMaterialId: number | null
}


export function resolveWorkTitle(
  materials: ApiMaterial[],
): ResolvedWorkTitle {
  if (materials.length === 0) {
    return { title: '—', titleNative: null, sourceMaterialId: null }
  }

  const pick = pickPrimary(materials)
  const native = pickNative(materials)

  return {
    title:            pick?.title ?? '—',
    titleNative:      native,
    sourceMaterialId: pick?.id ?? null,
  }
}


function pickPrimary(materials: ApiMaterial[]): ApiMaterial | null {
  // English-publishing material first — the universal title every
  // viewer can search externally. `languages` is BCP-47 list; we
  // accept both bare 'en' and any 'en-*' variant.
  const en = materials.find((m) =>
    (m.languages ?? []).some((l) => l.toLowerCase().split(/[-_]/)[0] === 'en'),
  )
  if (en) return en

  // Material with `title_native` — its romanized `title` is the
  // canonical-name proxy when no English source is present.
  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  if (withNative) return withNative

  return materials[0]!
}


function pickNative(materials: ApiMaterial[]): string | null {
  for (const m of materials) {
    const n = (m.title_native ?? '').trim()
    if (n) return n
  }
  return null
}
