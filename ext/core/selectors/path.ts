// Build a CSS selector path from an Element. Chosen for *stability*
// against React/Tailwind class hashing, not minimality:
//
//   - id          — only when not numeric/hash-looking
//   - data-testid — most stable signal in modern apps
//   - tag + nth-of-type — last resort
//
// We deliberately avoid class names (Tailwind utilities are stable but
// styled-components / CSS Modules generate `.css-xj3k2` strings that
// rotate every build).

const HASHLIKE = /^(?:[A-Za-z0-9_-]{6,}|\d+)$/

export function getSelectorPath(el: Element): string {
  const parts: string[] = []
  let cur: Element | null = el

  while (cur && cur !== document.documentElement) {
    const id = cur.id
    if (id && !HASHLIKE.test(id)) {
      parts.unshift(`#${cssEscape(id)}`)
      break  // id is unique enough on its own
    }

    let part = cur.tagName.toLowerCase()
    const testId = cur.getAttribute('data-testid')
    if (testId) {
      part += `[data-testid="${cssEscape(testId)}"]`
    } else {
      const parent = cur.parentElement
      if (parent) {
        const siblings = [...parent.children].filter(s => s.tagName === cur!.tagName)
        if (siblings.length > 1) {
          const idx = siblings.indexOf(cur) + 1
          part += `:nth-of-type(${idx})`
        }
      }
    }
    parts.unshift(part)
    cur = cur.parentElement
  }
  return parts.join(' > ')
}

/** Resolve a previously-saved selector against the current document.
 *  Returns null when the selector no longer matches anything (site
 *  re-skinned, route changed shape) so the caller can re-prompt. */
export function resolveSelector(selector: string): Element | null {
  try {
    return document.querySelector(selector)
  } catch {
    return null
  }
}

// Browsers ship `CSS.escape` natively; this is the polyfill subset we
// need for ids and attribute values. Replace any character outside the
// safe range with `\HEX `.
function cssEscape(s: string): string {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(s)
  }
  return s.replace(/[^a-zA-Z0-9_-]/g, ch =>
    `\\${ch.charCodeAt(0).toString(16)} `,
  )
}
