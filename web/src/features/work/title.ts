// Work title — resolve the display string for a Work from its
// sibling materials, biased toward the viewer's reading language.
//
// Real use case: a Vietnamese reader opens a manga that the OTruyen
// scanlation team translated and the same series is also indexed on
// HappyMH as raw Japanese. They want to see the title THEY know
// (VI), not a romanized English version they'd have to translate
// back.
//
// Priority (highest first):
//
//   1. The material the user explicitly picked via `?src=`
//      (`activeMaterial`). Their choice is final — even if it's
//      a non-target-lang source, respect it.
//
//   2. Material whose `languages` covers the viewer's `target_lang`.
//      The common path: a VI viewer lands on the VI source's title.
//
//   3. Material with a `title_native` set — kanji / hangul anchors
//      identity when the previous tiers miss (rare).
//
//   4. `materials[0]` — oldest fallback so something always renders.
//
// The native subtitle (the small italic line under the h1) is the
// first material's `title_native` regardless of which one supplied
// the primary title — purely cosmetic, doesn't drive identity.
//
// Cross-viewer determinism: if a URL carries `?src=`, every viewer
// gets the same title. Without `?src=`, the chain falls through to
// the viewer's target_lang — that's the intent, not a bug. Sharing
// a `/w/<id>?src=<X>` link is the way to pin a specific source.

import type { ApiMaterial } from '@shared/api/api'


export interface ResolvedWorkTitle {
  title:       string
  titleNative: string | null
  /** Material the title came from. Exposed so callers can compose
   *  attribution UI (e.g. "title from OTruyen") if needed. */
  sourceMaterialId: number | null
}


export function resolveWorkTitle(
  materials:      ApiMaterial[],
  activeMaterial: ApiMaterial | null,
  targetLang:     string | null,
): ResolvedWorkTitle {
  if (materials.length === 0) {
    return { title: '—', titleNative: null, sourceMaterialId: null }
  }

  const pick =
       activeMaterial
    ?? pickByLang(materials, targetLang)
    ?? pickByNative(materials)
    ?? materials[0]!

  return {
    title:            pick.title,
    titleNative:      firstNative(materials),
    sourceMaterialId: pick.id,
  }
}


function pickByLang(
  materials: ApiMaterial[],
  lang:      string | null,
): ApiMaterial | null {
  if (!lang) return null
  const norm = lang.toLowerCase().split(/[-_]/)[0]
  if (!norm) return null
  return materials.find((m) =>
    (m.languages ?? []).some((l) => l.toLowerCase().split(/[-_]/)[0] === norm),
  ) ?? null
}


function pickByNative(materials: ApiMaterial[]): ApiMaterial | null {
  return materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  ) ?? null
}


function firstNative(materials: ApiMaterial[]): string | null {
  for (const m of materials) {
    const n = (m.title_native ?? '').trim()
    if (n) return n
  }
  return null
}
