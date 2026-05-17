// Selector engine for manifest endpoints. Designed to cover the
// three reference sources (HappyMH JSON, OTruyen JSON, MangaDex JSON
// with nested cover relationships) without needing a full JSONPath
// library.
//
// Grammar:
//
//   {css-selector}            HTML text content of first match
//   {css-selector}@attr       HTML attribute value
//   @attr                     attribute of current row root
//
//   script:json({var})        extract a JS variable from an inline
//                             <script> tag as JSON. Matches the first
//                             <script> whose textContent contains the
//                             variable name, then regex-extracts the
//                             JSON object/array that follows `var =`.
//                             e.g. `script:json(g_th)` extracts
//                             `g_th = $.parseJSON('{"1":"j,..."}')`.
//
//   script:match({regex})     return capture group 1 of `regex` from
//                             the first matching <script> textContent.
//                             Useful for scalar vars like page counts.
//
//   $.path.to.field           JSON node (single-value path)
//   $.list[*]                 JSON array (use with queryJsonAll)
//   $.list[*]@field.subfield  for each element in list, read
//                             `.field.subfield`. queryJsonOne returns
//                             the FIRST non-null; queryJsonAll
//                             returns ALL non-null values.
//   $.list[0].key             explicit index
//   @field.subfield           attribute path on current row root
//
//   sel1 || sel2 || sel3      pipeline fallback — evaluate left to
//                             right, return the FIRST non-empty
//                             value. Whitespace around `||` is
//                             trimmed; useful for "title.en or
//                             title.{any} or altTitles[*]@en".
//
// Anything more exotic should be expressed via per-row `extras` and
// `=template` composition in the manifest.

const FALLBACK_SEP = /\s*\|\|\s*/

function isEmpty(v: unknown): boolean {
  if (v == null) return true
  if (typeof v === 'string') return v.trim().length === 0
  if (Array.isArray(v)) return v.length === 0
  return false
}

// ── script:json / script:match helpers ────────────────────────────

/** Extract a JS variable value from inline <script> tags.
 *  `script:json(varName)` → parsed JSON object/array.
 *  `script:match(regex)`  → capture group 1 as string. */
function queryScriptSpecial(
  root:     Element | Document,
  selector: string,
): string | null {
  const jsonMatch  = selector.match(/^script:json\(([^)]+)\)$/)
  const regexMatch = selector.match(/^script:match\((.+)\)$/)

  if (!jsonMatch && !regexMatch) return null

  const scripts = Array.from(root.querySelectorAll('script'))

  if (jsonMatch) {
    const varName = jsonMatch[1]!.trim()
    for (const s of scripts) {
      const text = s.textContent ?? ''
      if (!text.includes(varName)) continue
      // Match: varName = <json> OR varName = $.parseJSON('<json>')
      // Capture the raw JSON object/array.
      const patterns = [
        new RegExp(String.raw`\b${varName}\s*=\s*(\{[\s\S]*?\})\s*[;,\n]`),
        new RegExp(String.raw`\b${varName}\s*=\s*(\[[\s\S]*?\])\s*[;,\n]`),
        // $.parseJSON('...') or JSON.parse('...')
        new RegExp(String.raw`\b${varName}\s*=\s*(?:\$\.parseJSON|JSON\.parse)\s*\(\s*['"](.+?)['"]\s*\)`),
      ]
      for (const re of patterns) {
        const m = re.exec(text)
        if (m?.[1]) {
          // Validate it's parseable JSON before returning.
          try { JSON.parse(m[1]); return m[1] } catch { /* try next */ }
        }
      }
    }
    return null
  }

  // script:match(regex)
  const userRegex = regexMatch![1]!.trim()
  let re: RegExp
  try { re = new RegExp(userRegex) } catch { return null }

  for (const s of scripts) {
    const text = s.textContent ?? ''
    const m = re.exec(text)
    if (m?.[1] != null) return m[1]
  }
  return null
}

export function queryHtmlOne(
  root: Element | Document, selector: string,
): string | null {
  for (const sel of selector.split(FALLBACK_SEP)) {
    const v = queryHtmlOneSingle(root, sel)
    if (!isEmpty(v)) return v
  }
  return null
}

function queryHtmlOneSingle(
  root: Element | Document, selector: string,
): string | null {
  // script:json / script:match — special forms
  if (selector.startsWith('script:')) return queryScriptSpecial(root, selector)

  const { sel, attr } = splitAttr(selector)
  const el = sel ? root.querySelector(sel) : (root as Element)
  if (!el) return null
  if (attr) return el.getAttribute(attr)
  return cleanText(el.textContent)
}

export function queryHtmlAll(
  root: Element | Document, selector: string,
): Element[] {
  return Array.from(root.querySelectorAll(selector))
}

export function queryJsonOne(root: unknown, selector: string): unknown {
  for (const sel of selector.split(FALLBACK_SEP)) {
    const v = queryJsonOneSingle(root, sel)
    if (!isEmpty(v)) return v
  }
  return null
}

/** Single (no-fallback) selector against a JSON node. Supports the
 *  same grammar as `evalJsonPath` plus a leading `@` shorthand for
 *  paths anchored at the row root. Examples:
 *
 *    @id                        — root.id
 *    @attributes.title.en       — root.attributes.title.en
 *    @attributes.title.*        — first non-empty value of attr.title
 *    @relationships[*]@key.x    — for each rel, key.x; first non-null */
function queryJsonOneSingle(root: unknown, selector: string): unknown {
  // Cut on first `@` (the boundary marker). The right-hand side may
  // contain another `@` for the "map array, pick attr" form.
  const trimmed = selector.trim()
  const at = trimmed.indexOf('@')
  if (at < 0) {
    return evalJsonPath(root, trimmed)
  }
  // Anchor side: `$.path` OR empty (row root).
  const anchorRaw = trimmed.slice(0, at).trim()
  const anchor = anchorRaw.length === 0 ? root : evalJsonPath(root, anchorRaw)
  if (anchor == null) return null

  // Attribute side. May contain nested `@`:
  //   `attributes.title.en`
  //   `relationships[*]@key.fileName`
  const attrExpr = trimmed.slice(at + 1).trim()
  if (!attrExpr) return anchor
  return resolveAttrExpr(anchor, attrExpr)
}

/** Resolve an attribute expression against an already-evaluated
 *  anchor. Supports dotted paths, `*` wildcard, `[*]@subkey` forms.
 *  When `anchor` is an array, maps over each element and returns the
 *  first non-empty result. */
function resolveAttrExpr(anchor: unknown, expr: string): unknown {
  if (Array.isArray(anchor)) {
    for (const item of anchor) {
      const r = resolveAttrExpr(item, expr)
      if (!isEmpty(r)) return r
    }
    return null
  }
  // Split on a NESTED `@` (the row-vs-attr boundary inside the attr
  // side — used for arrays). Take first occurrence.
  const at = expr.indexOf('@')
  if (at < 0) {
    // Plain dotted path. Reuse evalJsonPath by routing through `$.`.
    return evalJsonPath(anchor, '$.' + expr)
  }
  const left  = expr.slice(0, at).trim()
  const right = expr.slice(at + 1).trim()
  const v = evalJsonPath(anchor, '$.' + left)
  if (Array.isArray(v)) {
    for (const item of v) {
      const r = resolveAttrExpr(item, right)
      if (!isEmpty(r)) return r
    }
    return null
  }
  return resolveAttrExpr(v, right)
}

/** List form. Falls back through `||` candidates until one yields
 *  a non-empty array. */
export function queryJsonAll(root: unknown, selector: string): unknown[] {
  for (const sel of selector.split(FALLBACK_SEP)) {
    const v = queryJsonAllSingle(root, sel)
    if (v.length > 0) return v
  }
  return []
}

function queryJsonAllSingle(root: unknown, selector: string): unknown[] {
  const trimmed = selector.trim()
  const at = trimmed.indexOf('@')
  if (at < 0) {
    // Pure JSONPath. Append `[*]` if not already a wildcard end.
    const sel = trimmed.endsWith('[*]') || trimmed.endsWith('.*')
      ? trimmed
      : trimmed + '[*]'
    const v = evalJsonPath(root, sel)
    return Array.isArray(v) ? v.filter((x) => x != null) : []
  }
  // `anchor@expr` — anchor must be array; map each item through expr.
  const anchorRaw = trimmed.slice(0, at).trim()
  const expr = trimmed.slice(at + 1).trim()
  const anchor = anchorRaw.length === 0 ? root : evalJsonPath(root, anchorRaw)
  if (!Array.isArray(anchor)) return []
  return anchor
    .map((item) => resolveAttrExpr(item, expr))
    .filter((x) => x != null)
}

// ── helpers ────────────────────────────────────────────────────────

function splitAttr(s: string): { sel: string; attr: string | null } {
  // `@attr` is the boundary between selector and attribute. JSONPaths
  // never embed `@`, HTML attribute names never embed `.`. We split
  // on the first `@` AFTER the last `.` or `[`, which handles both
  // `$.data.relationships[*]@attributes.fileName` and `img@data-src`.
  const trimmed = s.trim()
  const at = trimmed.indexOf('@')
  if (at < 0) return { sel: trimmed, attr: null }
  return {
    sel:  trimmed.slice(0, at).trim(),
    attr: trimmed.slice(at + 1).trim() || null,
  }
}

function cleanText(s: string | null | undefined): string | null {
  if (!s) return null
  const out = s.replace(/\s+/g, ' ').trim()
  return out.length > 0 ? out : null
}

/** Minimal JSONPath: `$`, `.key`, `.*`, `[index]`, `[*]`. `.*` and
 *  `[*]` both expand to "all values"; if used at the END of the path
 *  they return the FIRST non-empty entry (common case: `title.en`
 *  fallback to `title.*` for any language). When followed by more
 *  segments, the wildcard returns the array mapped through the
 *  remaining path.
 *
 *  This matches MangaDex's `title: { "<lang>": "..." }` and similar
 *  shapes without forcing the manifest to know the keys. */
function evalJsonPath(root: unknown, path: string): unknown {
  if (!path.startsWith('$')) return null
  let cur: unknown = root
  let i = 1
  while (i < path.length && cur != null) {
    const c = path[i]
    if (c === '.') {
      // `.*` — expand to all values of the current object.
      if (path[i + 1] === '*') {
        if (cur && typeof cur === 'object' && !Array.isArray(cur)) {
          cur = Object.values(cur)
        } else if (!Array.isArray(cur)) {
          return null
        }
        const rest = path.slice(i + 2)
        if (!rest) return firstNonEmpty(cur as unknown[])
        const mapped = (cur as unknown[]).map((item) =>
          evalJsonPath(item, `$${rest}`),
        )
        // Flatten one level when the next segment is [*] or .* —
        // handles $.titleListMap.*[*] where each object value is an
        // array and the caller wants a single merged flat list.
        if (rest.startsWith('[*]') || rest.startsWith('.*')) {
          return (mapped as unknown[][]).flat()
        }
        return mapped
      }
      const m = /^\.([A-Za-z_][\w-]*)/.exec(path.slice(i))
      if (!m) return null
      cur = (cur as Record<string, unknown>)[m[1]!]
      i += m[0].length
    } else if (c === '[') {
      const close = path.indexOf(']', i)
      if (close < 0) return null
      const idx = path.slice(i + 1, close).trim()
      if (idx === '*') {
        if (!Array.isArray(cur)) {
          if (cur && typeof cur === 'object') cur = Object.values(cur)
          else return []
        }
        const rest = path.slice(close + 1)
        if (!rest) return cur
        return (cur as unknown[]).map((item) =>
          evalJsonPath(item, `$${rest}`),
        )
      }
      // Equality filter: `[?key=value]` keeps array items whose
      // `.key` (single hop, no dots) equals the literal string
      // `value`. Drops items where the field is missing. Returns
      // the first surviving item (or its sub-projection when more
      // path segments follow). Quotes around the value are stripped.
      if (idx.startsWith('?')) {
        const eq = idx.indexOf('=', 1)
        if (eq < 0) return null
        const key = idx.slice(1, eq).trim()
        let val   = idx.slice(eq + 1).trim()
        if ((val.startsWith("'") && val.endsWith("'"))
            || (val.startsWith('"') && val.endsWith('"'))) {
          val = val.slice(1, -1)
        }
        if (!Array.isArray(cur)) return null
        const kept = (cur as unknown[]).filter((item) => {
          if (item == null || typeof item !== 'object') return false
          const v = (item as Record<string, unknown>)[key]
          return v != null && String(v) === val
        })
        const rest = path.slice(close + 1)
        if (!rest) return kept.length > 0 ? kept[0] : null
        // Trailing path segments — branch on whether they start with
        // a wildcard (project across every kept item) or a scalar
        // step (project from the first match only).
        const wildcard = rest.startsWith('[*]') || rest.startsWith('.*')
        if (wildcard) {
          return kept.map((item) => evalJsonPath(item, `$${rest}`))
        }
        if (kept.length === 0) return null
        return evalJsonPath(kept[0], `$${rest}`)
      }
      const n = Number(idx)
      if (!Array.isArray(cur) || !Number.isInteger(n)) return null
      cur = cur[n]
      i = close + 1
    } else {
      return null
    }
  }
  return cur
}

function firstNonEmpty(arr: unknown[]): unknown {
  for (const v of arr) {
    if (v == null) continue
    if (typeof v === 'string' && v.trim().length === 0) continue
    if (Array.isArray(v) && v.length === 0) continue
    return v
  }
  return null
}
