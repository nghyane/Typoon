/**
 * Deterministic noise classifier — port of typoon/stages/noise.py.
 *
 * `isAutoSkip(text)` returns true for non-diegetic chrome / watermarks /
 * page counters / pure-OCR rubble. Caller emits a zero-cost
 * `kind:"skip"` op for matching bubbles without an LLM round trip.
 *
 * Patterns live in noise_terms.txt (same file the Python pipeline uses).
 * Bundled as raw text via esbuild loader.
 *
 * `stripNoiseTokens(text)` removes inline noise tokens (domains, logo
 * emoji, scanlation tags) from otherwise-valid dialogue.
 */

// @ts-ignore — esbuild text loader
import NOISE_TERMS_TXT from "./noise_terms.txt";

// Pure punctuation/digit rubble (no letters in any script).
// \W in JS is ASCII-only — needs explicit Unicode-aware exclusion of letters.
const OCR_NOISE_RE      = /^[^\p{L}]+$/u;
const INLINE_DOMAIN_RE  = /\b[a-z0-9][a-z0-9.\-]{1,30}\.(?:co|com|net|org|io|to|me|xyz|info|live|app|tw|hk|cn)(?:\/\S*)?\b/gi;
const LOGO_EMOJI_RE     = /(?<![^\s！？。，、])[\u{1F300}-\u{1F9FF}](?![^\s！？。，、])/gu;
const CREDIT_TAG_RE     = /[\[《«【「\(]\s*[a-z0-9][a-z0-9\s.\-]{1,30}\s*[\]》»】」\)]/gi;
const CURRENCY_CHARS    = new Set(":/%$¥₩€£".split(""));

let cachedPatterns: RegExp[] | null = null;
function noisePatterns(): RegExp[] {
  if (cachedPatterns !== null) return cachedPatterns;
  const out: RegExp[] = [];
  for (const raw of String(NOISE_TERMS_TXT).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    try {
      // Python `re` and JS RegExp accept the same syntax for these patterns
      // (anchored start/end, escaped dots, char classes). `re.IGNORECASE` → /i.
      out.push(new RegExp(line, "iu"));
    } catch { /* skip malformed lines */ }
  }
  cachedPatterns = out;
  return out;
}

/** Mirror of Python re.fullmatch — anchor both ends. */
function fullmatch(re: RegExp, s: string): boolean {
  const m = s.match(re);
  return !!m && m[0] === s && m.index === 0;
}

export function isAutoSkip(text: string): boolean {
  const s = text.trim();
  if (!s) return true;
  // Pure digits / single letter
  if (/^\d+$/.test(s)) return true;
  if (s.length === 1 && /\p{L}/u.test(s)) return true;
  // Pure punctuation / digit rubble (unless currency / time separators)
  if (fullmatch(OCR_NOISE_RE, s)) {
    let hasCurrency = false;
    for (const ch of s) if (CURRENCY_CHARS.has(ch)) { hasCurrency = true; break; }
    if (!hasCurrency) return true;
  }
  for (const pat of noisePatterns()) if (fullmatch(pat, s)) return true;
  return false;
}

export function stripNoiseTokens(text: string): string {
  let s = text.replace(INLINE_DOMAIN_RE, "");
  s = s.replace(LOGO_EMOJI_RE, "");
  s = s.replace(CREDIT_TAG_RE, "");
  s = s.replace(/[ \t]{2,}/g, " ");
  return s.trim();
}
