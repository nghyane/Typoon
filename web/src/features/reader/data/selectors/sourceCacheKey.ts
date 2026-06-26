// sourceCacheKey — stable identity for the cache pool.
//
// Same chapter + same active kind → same key. The pool can hold the
// source across reader chrome/settings changes.

import type { ActiveSource } from '../types'


export function sourceCacheKey(
  workId:     string,
  chapterRef: string,
  active:     ActiveSource,
): string {
  switch (active.kind) {
    case 'none':
      return `${workId}:${chapterRef}:none`
    case 'raw-offline':
      return `${workId}:${chapterRef}:r-off`
    case 'raw-online':
      return `${workId}:${chapterRef}:r-${active.versionKey}`
  }
}
