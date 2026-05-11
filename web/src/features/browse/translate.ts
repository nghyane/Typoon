// Google widget translation — same endpoint translate.google.com's
// in-browser widget uses, no API key, free, ~50-150ms latency.
//
// Stability: this is an unauth endpoint. Google can change the
// protocol at any time (last big change: 2016). Every caller MUST
// treat null/throw as a graceful fallback to the original text — no
// user-visible error. The cache lives forever (text → translation
// is deterministic) so a temporary outage doesn't re-spam Google.
//
// Rate gate: 12 req/s client-side. Google's free tier rate-limits at
// roughly that range; exceeding triggers a 429 + IP-level cooldown.

import { pfetch } from './proxy'

const ENDPOINT  = 'https://translate.googleapis.com/translate_a/single'
const MAX_CHARS = 4500            // Google rejects > ~5000
const MIN_GAP   = 80              // ms — ~12 req/s

let lastCall = 0
async function gate(): Promise<void> {
  const wait = Math.max(0, MIN_GAP - (Date.now() - lastCall))
  if (wait) await new Promise((r) => setTimeout(r, wait))
  lastCall = Date.now()
}

/** Translate a single text. Returns null on empty input / network
 *  failure / unexpected response shape. */
export async function translate(
  text: string,
  target: string,
  source: string = 'auto',
): Promise<string | null> {
  const trimmed = text.trim()
  if (!trimmed) return null
  await gate()
  const url = new URL(ENDPOINT)
  url.searchParams.set('client', 'gtx')
  url.searchParams.set('sl', source)
  url.searchParams.set('tl', target)
  url.searchParams.set('dt', 't')
  url.searchParams.set('q',  trimmed)
  try {
    const r = await pfetch(url.toString())
    if (!r.ok) return null
    const data = await r.json() as unknown
    // Shape: [[[ "translated", "src", null, null, conf ], …], srcLang, …]
    if (!Array.isArray(data) || !Array.isArray(data[0])) return null
    const joined = (data[0] as unknown[])
      .map((seg) => Array.isArray(seg) && typeof seg[0] === 'string' ? seg[0] : '')
      .join('')
      .trim()
    return joined || null
  } catch { return null }
}

// Sentinel between batched texts. We wrap each item in `<x>...</x>`
// because Google preserves raw HTML tags verbatim in translation
// output AND keeps each tag's content translated *independently*
// (so item N doesn't bleed into item N-1's grammar). Other sentinels
// we tried (`@@SEP@@`, double-newline, fancy unicode) all caused
// Google to translate each item in the context of its neighbours,
// producing mixed-language output.
const TAG_OPEN  = '<x>'
const TAG_CLOSE = '</x>'

/** Batch translate. Wraps each input in `<x>` so Google translates
 *  every item independently (otherwise neighbouring items bleed into
 *  one another's grammar and produce mixed-language output). On
 *  count mismatch falls back to per-item parallel calls. Chunked
 *  when total payload would exceed Google's char limit. */
export async function translateBatch(
  texts: string[],
  target: string,
  source: string = 'auto',
): Promise<(string | null)[]> {
  if (texts.length === 0) return []
  const wrapped = texts.map((t) => TAG_OPEN + t + TAG_CLOSE).join('')
  if (wrapped.length > MAX_CHARS) {
    return chunkedBatch(texts, target, source)
  }
  const out = await translate(wrapped, target, source)
  if (!out) return texts.map(() => null)
  const parts = parseWrapped(out)
  if (parts.length !== texts.length) {
    return Promise.all(texts.map((t) => translate(t, target, source)))
  }
  return parts
}

function parseWrapped(s: string): string[] {
  const out: string[] = []
  const re = /<x>([\s\S]*?)<\/x>/g
  let m: RegExpExecArray | null
  while ((m = re.exec(s)) !== null) {
    out.push((m[1] ?? '').trim())
  }
  return out
}

async function chunkedBatch(
  texts: string[], target: string, source: string,
): Promise<(string | null)[]> {
  const wrap = (t: string) => TAG_OPEN + t + TAG_CLOSE
  const chunks: string[][] = []
  let cur: string[] = []
  let len = 0
  for (const t of texts) {
    const tlen = wrap(t).length
    if (len + tlen > MAX_CHARS && cur.length > 0) {
      chunks.push(cur)
      cur = []
      len = 0
    }
    cur.push(t)
    len += tlen
  }
  if (cur.length > 0) chunks.push(cur)
  const out: (string | null)[] = []
  for (const c of chunks) out.push(...await translateBatch(c, target, source))
  return out
}
