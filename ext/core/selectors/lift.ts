// Smart-lift: when the user clicks a single <img> (or a thin wrapper
// around one), walk up the DOM until we find an ancestor that
// contains multiple `<img>` siblings of similar size.
//
// Heuristic: stop at the smallest ancestor whose <img> count is ≥
// `minSiblings` (default 2) AND whose imgs share roughly the same
// aspect ratio as the original click. The aspect-ratio check avoids
// lifting from "page reader" into "page reader + sidebar thumbnail
// gallery + ads".
//
// Falls back to `el` itself if no better ancestor is found within
// `maxHops`.

export interface LiftOptions {
  minSiblings?: number
  maxHops?:     number
  /** Allow up to ±this fraction difference in aspect ratio between
   *  the click target and a sibling for it to count as "similar".
   *  Default 0.5 — generous, since manga pages can vary 0.6–1.5x. */
  aspectTolerance?: number
}

export function smartLift(el: Element, opts: LiftOptions = {}): Element {
  const minSiblings     = opts.minSiblings     ?? 2
  const maxHops         = opts.maxHops         ?? 6
  const aspectTolerance = opts.aspectTolerance ?? 0.5

  const baseRect   = el.getBoundingClientRect()
  const baseAspect = baseRect.width > 0 && baseRect.height > 0
    ? baseRect.width / baseRect.height
    : 1

  let cur: Element | null = el
  for (let i = 0; i < maxHops; i++) {
    const parent = cur.parentElement
    if (!parent || parent === document.documentElement) break

    const imgs = parent.querySelectorAll<HTMLImageElement>('img')
    if (imgs.length >= minSiblings) {
      const similar = countSimilar(imgs, baseAspect, aspectTolerance)
      if (similar >= minSiblings) return parent
    }
    cur = parent
  }
  return el
}

function countSimilar(
  imgs: NodeListOf<HTMLImageElement>,
  baseAspect: number,
  tolerance: number,
): number {
  let count = 0
  for (const img of imgs) {
    const r = img.getBoundingClientRect()
    if (r.width < 32 || r.height < 32) continue   // skip tiny icons
    const aspect = r.width / r.height
    if (Math.abs(aspect - baseAspect) / baseAspect <= tolerance) count++
  }
  return count
}
