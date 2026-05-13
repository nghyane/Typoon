// Chapter-number normalisation — declarative, no JS eval.
//
// The manifest supplies a `ChapterNumberNorm` spec; this module
// compiles it into a deterministic string→string transform that the
// runtime applies after extracting each chapter row. Output is the
// `numberNorm` field on `MangaChapterRef`, which the server later
// stores as `work_chapters.number_norm`.
//
// Two sources of the same manga that publish chapter 40 with
// different display strings ("Chương 040" vs "Chapter 40") must
// produce the same `numberNorm` so the cross-source chapter list
// collapses to one row.
//
// Evaluation:
//
//   1. Pick input from `spec.input` ('number' default, or 'label').
//   2. Try regex `patterns` top-down. First match wins; capture
//      group 1 (if present) becomes the extracted string, otherwise
//      the whole match.
//   3. No match → apply `spec.default` ('slug' default):
//        slug      lowercase + trim of raw input
//        empty     "" (treats unparseable labels as one shared row)
//        verbatim  raw input untouched
//   4. Run `spec.postprocess` steps in order on the resulting
//      string. Available: 'lowercase', 'trim', 'stripLeadingZeros'.
//
// Compiled regexes are cached per-spec so a manifest with hundreds
// of chapters doesn't re-parse the same patterns N times.

import type { ChapterNumberNorm } from './types'


/** Global fallback when neither the manifest nor the endpoint
 *  declares a normaliser. Designed to cover the common case across
 *  three reference sources (HappyMH zh, OTruyen vi, MangaDex multi):
 *
 *    "Chapter 40 — Showdown"  → "40"
 *    "Chương 040"             → "40"      (postprocess strips zeros)
 *    "40.5"                   → "40.5"
 *    "第106话"                 → "106"
 *    "40화"                    → "40"
 *    "Extra"                  → "extra"   (default=slug)
 *
 *  Sources with chapter-number quirks beyond this list should
 *  declare their own `chapterNumberNorm` to win the dedup. */
export const DEFAULT_CHAPTER_NUMBER_NORM: ChapterNumberNorm = {
  input: 'number',
  patterns: [
    // Latin chapter prefixes + decimal number.
    'Ch(?:apter|ương|\\.|)\\s*:?\\s*([0-9]+(?:\\.[0-9]+)?)',
    // CJK chapter markers — emit just the digits, ignore the suffix.
    '第\\s*([0-9]+(?:\\.[0-9]+)?)\\s*[话話回]?',
    '([0-9]+(?:\\.[0-9]+)?)\\s*화',
    // Last-resort: first numeric run anywhere in the string.
    '([0-9]+(?:\\.[0-9]+)?)',
  ],
  default: 'slug',
  postprocess: ['stripLeadingZeros', 'lowercase'],
}


// ─── compilation ──────────────────────────────────────────────────


interface CompiledNorm {
  input:       'number' | 'label'
  patterns:    RegExp[]
  default:     'slug' | 'empty' | 'verbatim'
  postprocess: ('lowercase' | 'trim' | 'stripLeadingZeros')[]
}


const cache = new WeakMap<ChapterNumberNorm, CompiledNorm>()


export function compileChapterNumberNorm(
  spec: ChapterNumberNorm | undefined | null,
): CompiledNorm {
  const effective = spec ?? DEFAULT_CHAPTER_NUMBER_NORM
  const cached = cache.get(effective)
  if (cached) return cached
  const compiled: CompiledNorm = {
    input:    effective.input    ?? 'number',
    default:  effective.default  ?? 'slug',
    patterns: (effective.patterns ?? []).map(compilePattern),
    postprocess: effective.postprocess ?? ['stripLeadingZeros', 'lowercase'],
  }
  cache.set(effective, compiled)
  return compiled
}


function compilePattern(pat: string): RegExp {
  // Per-pattern flags: case-insensitive helps for Latin prefixes
  // without forcing every source to lowercase explicitly. CJK
  // patterns are unaffected by /i. No /g — we want first match only.
  try {
    return new RegExp(pat, 'i')
  } catch (err) {
    throw new Error(
      `chapterNumberNorm pattern invalid: ${JSON.stringify(pat)} (${(err as Error).message})`,
    )
  }
}


// ─── application ──────────────────────────────────────────────────


/** Apply the compiled norm to a chapter row's raw fields. The
 *  `fields` shape matches what `runtime.buildChapter` has already
 *  resolved — only the bits we need (`number`, `label`). */
export function applyChapterNumberNorm(
  compiled: CompiledNorm,
  fields: { number: string; label: string | null },
): string {
  const raw = compiled.input === 'label'
    ? (fields.label ?? '')
    : fields.number
  return runNorm(compiled, raw)
}


function runNorm(compiled: CompiledNorm, raw: string): string {
  let extracted: string | null = null
  for (const re of compiled.patterns) {
    const m = re.exec(raw)
    if (m) {
      extracted = (m[1] ?? m[0]) ?? null
      if (extracted != null) break
    }
  }

  let out: string
  if (extracted != null) {
    out = extracted
  } else {
    switch (compiled.default) {
      case 'slug':     out = raw; break
      case 'empty':    out = ''; break
      case 'verbatim': out = raw; break
    }
  }

  for (const step of compiled.postprocess) {
    switch (step) {
      case 'lowercase':
        out = out.toLowerCase()
        break
      case 'trim':
        out = out.trim()
        break
      case 'stripLeadingZeros':
        // Strip zeros only on the integer portion. "040.5" → "40.5",
        // "0.5" → "0.5" (keep the leading 0 because the integer is 0).
        out = out.replace(/^0+(?=\d)/, '')
        break
    }
  }

  // Final cosmetic trim — every spec gets it free of charge so
  // patterns that leak whitespace around their capture still
  // produce a clean key.
  return out.trim()
}


// ─── exported for unit-test convenience ───────────────────────────


export function _unsafeTestNormalize(
  raw: string,
  spec?: ChapterNumberNorm,
): string {
  return runNorm(compileChapterNumberNorm(spec), raw)
}
