// resolveSource — pure decision: pref + chapter sources → ActiveSource.
//
// Layering:
//   1. If pref explicitly picks raw → use raw (skip saved archives)
//   2. If pref is auto → saved archive → raw (target lang version,
//      offline blob > online URLs)
//
// `needsRawUrls` reports whether the caller must probe raw page
// URLs for the chosen path. The reader uses this to gate the
// manifest fetch — saved archives don't need raw URLs at all.

import { pickBestVersion } from '@features/work/data/selectors/mergeChapters'
import type { MergedChapter, SourceVersion } from '@features/work/data/types'

import type {
  ActiveSource, ChapterSources, SourcePref,
} from '../types'


export function versionKeyOf(v: SourceVersion): string {
  return `${v.source.manifest.id}:${v.ref.id}`
}


/** Resolution context — what the caller knows ahead of any
 *  manifest fetch. `rawUrls` is empty until the probe lands. */
export interface ResolveInput {
  pref:       SourcePref
  chapter:    MergedChapter | null
  sources:    ChapterSources
  targetLang: string
  /** Raw page URLs probe result. Empty array = not yet fetched OR
   *  the chosen version has no readable raw. */
  rawUrls:    string[]
}


export interface ResolveOutput {
  active:        ActiveSource
  /** True when the resolver needs raw URLs to finish — caller
   *  should fetch + retry. Lets the reader skip manifest probes
   *  for saved-archive paths. */
  needsRawUrls:  boolean
  /** The version the resolver picked (if raw-online path). Useful
   *  for downstream consumers like the picker. */
  pickedVersion: SourceVersion | null
}


export function resolveSource(input: ResolveInput): ResolveOutput {
  const { pref, chapter, sources, targetLang, rawUrls } = input
  const { saved, versions } = sources

  // Explicit raw pick — skip saved archive fallback entirely.
  //
  // Note: we DO NOT fall back to `saved.kind === 'raw'` here, even
  // when an offline raw blob exists for this chapter. The blob is
  // keyed by `${work_id}:${chapter_ref}` only — it doesn't carry the
  // versionKey of the source it was saved from. If the user explicitly
  // picks a different raw version, returning the existing blob would
  // silently ignore their pick (page count and image content would be
  // from the wrong source). The auto path below still uses saved-raw
  // as a fast offline fallback.
  if (pref.kind === 'raw') {
    const picked = chapter
      ? versions.find(v => versionKeyOf(v) === pref.versionKey)
      : null
    if (picked) {
      if (rawUrls.length === 0) {
        return {
          active: { kind: 'none' },
          needsRawUrls: true,
          pickedVersion: picked,
        }
      }
      return {
        active: {
          kind:       'raw-online',
          versionKey: pref.versionKey,
          urls:       rawUrls,
        },
        needsRawUrls: false,
        pickedVersion: picked,
      }
    }
    // Pref points to a missing version — fall through to auto.
  }

  // Auto mode: saved archive → raw (best version)
  if (saved?.kind === 'raw') {
    return {
      active: { kind: 'raw-offline', archiveId: saved.id, blob: saved.blob },
      needsRawUrls: false,
      pickedVersion: null,
    }
  }
  if (!chapter || versions.length === 0) {
    return { active: { kind: 'none' }, needsRawUrls: false, pickedVersion: null }
  }
  const best = pickBestVersion(chapter, targetLang.toLowerCase())
  if (!best) {
    return { active: { kind: 'none' }, needsRawUrls: false, pickedVersion: null }
  }
  if (rawUrls.length === 0) {
    return {
      active: { kind: 'none' },
      needsRawUrls: true,
      pickedVersion: best,
    }
  }
  return {
    active: {
      kind:       'raw-online',
      versionKey: versionKeyOf(best),
      urls:       rawUrls,
    },
    needsRawUrls: false,
    pickedVersion: best,
  }
}
