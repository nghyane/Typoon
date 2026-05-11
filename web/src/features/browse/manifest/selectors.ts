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
        return (cur as unknown[]).map((item) =>
          evalJsonPath(item, `$${rest}`),
        )
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
