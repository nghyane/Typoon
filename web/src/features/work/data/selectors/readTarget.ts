// Read target — what does the primary CTA open?
//
// Pure function: history ∪ chapter list → either resume-position or
// first chapter. Returns null when no chapter is available.

import type { HistoryItem, MergedChapter } from '../types'


export interface ReadTarget {
  ref:      string
  isResume: boolean
}


export function pickReadTarget(
  historyMap: ReadonlyMap<string, HistoryItem>,
  merged:     readonly MergedChapter[],
): ReadTarget | null {
  // Resume = most recently-read across all chapters
  let resume: HistoryItem | null = null
  for (const h of historyMap.values()) {
    if (!resume || h.last_read_at > resume.last_read_at) resume = h
  }
  if (resume) return { ref: resume.chapter_ref, isResume: true }

  // First chapter = lowest sortKey
  let first: MergedChapter | null = null
  for (const ch of merged) {
    if (!first || ch.sortKey < first.sortKey) first = ch
  }
  return first ? { ref: first.numberNorm, isResume: false } : null
}
