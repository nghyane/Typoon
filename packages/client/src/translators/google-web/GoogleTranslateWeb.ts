import type { TranslatedUnit } from '../../domain/translation'
import type { Translator } from '../translator'
import { AsyncLimiter } from '../../flow/AsyncLimiter'
import { batchUnits, parseTranslatedBatch, serializeBatch, toTranslatedUnit } from '../../pipeline/translation/MarkerProtocol'

export interface GoogleTranslateWebOptions {
  readonly endpoint?: string
  readonly maxBatchChars?: number
  readonly maxConcurrency?: number
}

const DEFAULT_MAX_BATCH_CHARS = 2_000
const DEFAULT_MAX_CONCURRENCY = 8

export class GoogleTranslateWeb implements Translator {
  readonly name = 'google-translate-web'
  private readonly endpoint: string
  private readonly maxBatchChars: number
  private readonly limiter: AsyncLimiter

  constructor(options: GoogleTranslateWebOptions = {}) {
    this.endpoint = options.endpoint ?? 'https://translate.googleapis.com/translate_a/single'
    this.maxBatchChars = options.maxBatchChars ?? DEFAULT_MAX_BATCH_CHARS
    this.limiter = new AsyncLimiter(options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY)
  }

  async translateUnits({ units, sourceLang, targetLang, signal }: Parameters<Translator['translateUnits']>[0]): Promise<readonly TranslatedUnit[]> {
    const batches = batchUnits(units, this.maxBatchChars)
    const resultBatches = new Array<readonly string[]>(batches.length)

    await Promise.all(batches.map((batch, bi) =>
      this.limiter.run(async () => {
        if (batch.every(unit => !unit.sourceText.trim())) {
          resultBatches[bi] = batch.map(() => '')
          return
        }
        const translated = await translateText({
          endpoint: this.endpoint,
          sourceText: serializeBatch(batch),
          sourceLang, targetLang, signal,
        })
        resultBatches[bi] = parseTranslatedBatch(translated, batch.length)
      }),
    ))

    const out: TranslatedUnit[] = []
    for (let bi = 0; bi < batches.length; bi++) {
      const batch = batches[bi]!
      const results = resultBatches[bi]!
      for (let j = 0; j < batch.length; j++) {
        out.push(toTranslatedUnit(batch[j]!, results[j]))
      }
    }
    return out
  }

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
  return parsePayload(await res.json() as unknown)
}

function parsePayload(payload: unknown): string {
  if (!Array.isArray(payload) || !Array.isArray(payload[0])) {
    throw new Error('unexpected google translate payload')
  }
  return payload[0]
    .map(part => Array.isArray(part) ? String(part[0] ?? '') : '')
    .join('')
    .trim()
}

function normalizeLang(lang: string | null): string | null {
  if (!lang) return null
  const lower = lang.toLowerCase()
  if (lower.startsWith('zh-hant') || lower === 'zh-tw') return 'zh-TW'
  if (lower.startsWith('zh')) return 'zh-CN'
  return lower.split('-')[0] ?? lower
}
