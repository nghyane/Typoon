// Walk a container element and extract every image URL it contains, in
// DOM order, deduplicated. Handles the lazy-load patterns common to
// manga viewers:
//
//   <img src>             — eager
//   <img srcset>          — pick largest descriptor
//   <img data-src>        — common lazy attribute
//   <img data-original>   — older lazy libs
//   <img data-lazy-src>   — slick / lozad
//   <picture><source>     — responsive
//   style: background-image: url(...)  — background galleries

export interface ImageRef {
  /** Absolute URL the page would load. Already URL-resolved. */
  url:    string
  width?:  number
  height?: number
}

const LAZY_ATTRS = ['data-src', 'data-original', 'data-lazy-src', 'data-srcset']

export function extractImages(root: Element): ImageRef[] {
  const seen = new Set<string>()
  const out:  ImageRef[] = []

  const push = (raw: string | null | undefined, w?: number, h?: number) => {
    if (!raw) return
    const url = absolutise(raw.trim())
    if (!url || seen.has(url)) return
    seen.add(url)
    out.push({ url, width: w, height: h })
  }

  // <img>
  for (const el of root.querySelectorAll<HTMLImageElement>('img')) {
    const fromSrcset = bestFromSrcset(el.srcset)
    if (fromSrcset) {
      push(fromSrcset, el.naturalWidth || undefined, el.naturalHeight || undefined)
      continue
    }
    if (el.src && !el.src.startsWith('data:')) {
      push(el.src, el.naturalWidth || undefined, el.naturalHeight || undefined)
      continue
    }
    for (const attr of LAZY_ATTRS) {
      const v = el.getAttribute(attr)
      if (!v) continue
      const best = attr === 'data-srcset' ? bestFromSrcset(v) : v
      if (best) {
        push(best)
        break
      }
    }
  }

  // <picture><source>
  for (const el of root.querySelectorAll<HTMLSourceElement>('picture source')) {
    const best = bestFromSrcset(el.srcset)
    if (best) push(best)
  }

  // background-image: url(...)
  for (const el of root.querySelectorAll<HTMLElement>('[style*="background-image"]')) {
    const m = /url\(\s*['"]?([^'")]+)['"]?\s*\)/.exec(el.style.backgroundImage)
    if (m) push(m[1])
  }

  return out
}

/** Pick the highest-resolution candidate from a srcset string.
 *  Format: `url1 1x, url2 2x` or `url1 480w, url2 1024w`. */
function bestFromSrcset(srcset: string | null | undefined): string | null {
  if (!srcset) return null
  let bestUrl: string | null = null
  let bestRank = -Infinity
  for (const part of srcset.split(',')) {
    const trimmed = part.trim()
    if (!trimmed) continue
    const [url, descriptor] = trimmed.split(/\s+/, 2)
    if (!url) continue
    let rank = 1
    if (descriptor) {
      const m = /^(\d+(?:\.\d+)?)([wx])$/.exec(descriptor)
      if (m) rank = parseFloat(m[1]!)
    }
    if (rank > bestRank) {
      bestRank = rank
      bestUrl  = url
    }
  }
  return bestUrl
}

function absolutise(url: string): string {
  if (!url) return ''
  if (url.startsWith('data:')) return ''
  try {
    return new URL(url, document.baseURI).href
  } catch {
    return ''
  }
}

/** Best-effort: nudge IntersectionObserver-based lazy loaders without
 *  scrolling the user's view. Triggers `loadeager` on `<img loading="lazy">`
 *  by reading their bounding rect (forces layout) and dispatches a
 *  scroll event so listeners with throttle/debounce fire.
 *
 *  Does NOT change `window.scrollY` — picker overlay must not jerk
 *  the user's reading position around. For deep lazy-loaded
 *  viewers, callers should use `scrollPrimeAll` (which saves and
 *  restores scroll). */
export async function primeLazyLoad(_root: Element, ms = 50): Promise<void> {
  // Force layout pass — some lazy libs (e.g. lazysizes) react to
  // any layout read by re-evaluating visible images.
  void document.body.offsetHeight
  // Some libraries listen for scroll events to re-check visibility.
  window.dispatchEvent(new Event('scroll'))
  await new Promise(r => setTimeout(r, ms))
}

/** Aggressive lazy-load primer — scrolls through the container in
 *  viewport-sized steps so every lazy-loaded image inside fires its
 *  IntersectionObserver. Necessary on long manga viewers (200+
 *  pages) where only the first few are loaded by default.
 *
 *  Saves and restores the user's scroll position so they end up
 *  exactly where they were before. Returns when the container's
 *  image count stops growing — the caller can rely on a stable
 *  extract right after. */
export async function scrollPrimeAll(
  root: Element,
  opts: { stepMs?: number; maxSteps?: number } = {},
): Promise<void> {
  const stepMs   = opts.stepMs   ?? 200
  const maxSteps = opts.maxSteps ?? 80    // ~80 viewport-heights cap

  const savedX = window.scrollX
  const savedY = window.scrollY

  try {
    // Start at the top.
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' as ScrollBehavior })
    await new Promise(r => setTimeout(r, stepMs))

    let lastCount = -1
    let stableCycles = 0
    for (let i = 0; i < maxSteps; i++) {
      window.scrollBy({ top: window.innerHeight * 0.9, behavior: 'instant' as ScrollBehavior })
      await new Promise(r => setTimeout(r, stepMs))

      const count = root.querySelectorAll('img').length
      if (count === lastCount) {
        stableCycles++
        if (stableCycles >= 2) break
      } else {
        stableCycles = 0
        lastCount = count
      }

      const atBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 4
      if (atBottom) break
    }
  } finally {
    // Always restore — even if we threw mid-scroll the user shouldn't
    // be left at the bottom of a 200-page viewer.
    window.scrollTo({ top: savedY, left: savedX, behavior: 'instant' as ScrollBehavior })
  }
}
