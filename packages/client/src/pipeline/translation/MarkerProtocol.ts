import type { TranslatedUnit, TranslationUnit } from '../../domain/translation'

export function batchUnits(units: readonly TranslationUnit[], maxBatchChars: number): TranslationUnit[][] {
  const batches: TranslationUnit[][] = []
  let current: TranslationUnit[] = []
  let currentChars = 0
  for (const unit of units) {
    if (!unit.sourceText.trim()) {
      if (current.length) {
        batches.push(current)
        current = []
        currentChars = 0
      }
      batches.push([unit])
      continue
    }
    const nextChars = unit.sourceText.length + unit.id.length + 18
    if (current.length && currentChars + nextChars > maxBatchChars) {
      batches.push(current)
      current = []
      currentChars = 0
    }
    current.push(unit)
    currentChars += nextChars
  }
  if (current.length) batches.push(current)
  return batches
}

export function serializeBatch(units: readonly TranslationUnit[]): string {
  return units
    .map(unit => `${marker(unit.id)}\n${normalizeSourceText(unit.sourceText)}`)
    .join('\n')
}

export function normalizeSourceText(sourceText: string): string {
  return sourceText
    .trim()
    .replace(/([\p{L}\p{N}])-\s*\n\s*([\p{L}\p{N}])/gu, '$1$2')
    .replace(/\s*\n+\s*/gu, ' ')
    .replace(/[ \t]+/gu, ' ')
    .trim()
}

export function parseTranslatedBatch(text: string): Map<string, string> {
  const out = new Map<string, string>()
  const matches = [...text.matchAll(/^@@TYPOON_ID:([^@\n]+)@@\s*$/gmu)]
  for (let i = 0; i < matches.length; i++) {
    const match = matches[i]!
    const id = match[1]
    if (!id) continue
    const start = (match.index ?? 0) + match[0].length
    const end = matches[i + 1]?.index ?? text.length
    out.set(id, text.slice(start, end).trim())
  }
  return out
}

function marker(id: string): string {
  return `@@TYPOON_ID:${id}@@`
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
