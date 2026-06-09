import type { TranslatedUnit, TranslationUnit } from '../../domain/translation'
import type { Translator } from '../translator'

export interface GoogleTranslateWebOptions {
  readonly endpoint?: string
  readonly maxBatchChars?: number
}

export class GoogleTranslateWeb implements Translator {
  readonly name = 'google-translate-web'
  private readonly endpoint: string
  private readonly maxBatchChars: number

  constructor(options: GoogleTranslateWebOptions = {}) {
    this.endpoint = options.endpoint ?? 'https://translate.googleapis.com/translate_a/single'
    this.maxBatchChars = options.maxBatchChars ?? 2_000
  }

  async translateUnits({ units, sourceLang, targetLang, signal }: Parameters<Translator['translateUnits']>[0]): Promise<readonly TranslatedUnit[]> {
    const out: TranslatedUnit[] = []
    for (const batch of batchUnits(units, this.maxBatchChars)) {
      if (batch.every(unit => !unit.sourceText.trim())) {
        batch.forEach(unit => out.push(toTranslatedUnit(unit, '')))
        continue
      }
      const translated = await translateText({ endpoint: this.endpoint, sourceText: serializeBatch(batch), sourceLang, targetLang, signal })
      const byId = parseTranslatedBatch(translated)
      for (const unit of batch) out.push(toTranslatedUnit(unit, byId.get(unit.id) ?? ''))
    }
    return out
  }
}

function batchUnits(units: readonly TranslationUnit[], maxBatchChars: number): TranslationUnit[][] {
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

function serializeBatch(units: readonly TranslationUnit[]): string {
  return units
    .map(unit => `${marker(unit.id)}\n${unit.sourceText.trim()}`)
    .join('\n')
}

function parseTranslatedBatch(text: string): Map<string, string> {
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

async function translateText(args: {
  endpoint: string
  sourceText: string
  sourceLang: string | null
  targetLang: string
  signal?: AbortSignal
}): Promise<string> {
  const url = new URL(args.endpoint)
  url.searchParams.set('client', 'gtx')
  url.searchParams.set('sl', normalizeLang(args.sourceLang) ?? 'auto')
  url.searchParams.set('tl', normalizeLang(args.targetLang) ?? args.targetLang)
  url.searchParams.set('dt', 't')
  url.searchParams.set('q', args.sourceText)
  const res = await fetch(url, { signal: args.signal })
  if (!res.ok) throw new Error(`google translate failed: ${res.status}`)
  return parseGoogleTranslatePayload(await res.json() as unknown)
}

function parseGoogleTranslatePayload(payload: unknown): string {
  if (!Array.isArray(payload) || !Array.isArray(payload[0])) {
    throw new Error('unexpected google translate payload')
  }
  return payload[0]
    .map(part => Array.isArray(part) ? String(part[0] ?? '') : '')
    .join('')
    .trim()
}

function toTranslatedUnit(unit: TranslationUnit, targetText: string): TranslatedUnit {
  return {
    unitId: unit.id,
    placementId: unit.placementId,
    pageIndex: unit.pageIndex,
    kind: targetText.trim() ? unit.kind : 'skip',
    role: unit.role,
    sourceText: unit.sourceText,
    targetText,
  }
}

function normalizeLang(lang: string | null): string | null {
  if (!lang) return null
  const lower = lang.toLowerCase()
  if (lower.startsWith('zh-hant') || lower === 'zh-tw') return 'zh-TW'
  if (lower.startsWith('zh')) return 'zh-CN'
  return lower.split('-')[0] ?? lower
}
