import type { TranslatedUnit, TranslationUnit } from '../../domain/translation'

const MARKER_REGEX = /@(\d+)@/g
const MARKER_PREFIX_LENGTH = 4 // "@0@" = 3 chars + newline

export function batchUnits(units: readonly TranslationUnit[], maxBatchChars: number): TranslationUnit[][] {
  const batches: TranslationUnit[][] = []
  let current: TranslationUnit[] = []
  let currentChars = 0
  for (const unit of units) {
    if (!unit.sourceText.trim()) {
      if (current.length) { batches.push(current); current = []; currentChars = 0 }
      batches.push([unit])
      continue
    }
    const overhead = MARKER_PREFIX_LENGTH + unit.sourceText.length + unit.id.length + 1
    if (current.length && currentChars + overhead > maxBatchChars) {
      batches.push(current)
      current = []
      currentChars = 0
    }
    current.push(unit)
    currentChars += overhead
  }
  if (current.length) batches.push(current)
  return batches
}

export function serializeBatch(units: readonly TranslationUnit[]): string {
  return units
    .map((unit, i) => `@${i}@${normalizeSourceText(unit.sourceText)}`)
    .join('\n')
}

export function parseTranslatedBatch(text: string, expectedCount: number): string[] {
  const parts = new Array<string>(expectedCount)
  const matches = [...text.matchAll(MARKER_REGEX)]
  for (let i = 0; i < matches.length; i++) {
    const match = matches[i]!
    const index = parseInt(match[1]!, 10)
    const start = match.index! + match[0].length
    const end = matches[i + 1]?.index ?? text.length
    parts[index] = text.slice(start, end).trim()
  }
  // Fill any missing indices with empty string (translator may have dropped a marker)
  for (let i = 0; i < expectedCount; i++) {
    if (parts[i] === undefined) parts[i] = ''
  }
  return parts
}

export function toTranslatedUnit(unit: TranslationUnit, targetText: string): TranslatedUnit {
  return {
    unitId: unit.id,
    pageIndex: unit.pageIndex,
    kind: targetText.trim() ? unit.kind : 'skip',
    role: unit.role,
    sourceText: unit.sourceText,
    targetText,
  }
}

function normalizeSourceText(sourceText: string): string {
  return sourceText
    .trim()
    .replace(/([\p{L}\p{N}])-\s*\n\s*([\p{L}\p{N}])/gu, '$1$2')
    .replace(/\s*\n+\s*/gu, ' ')
    .replace(/[ \t]+/gu, ' ')
    .trim()
}
