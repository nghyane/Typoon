// Floating overlay for the picker wizard. Vanilla DOM in a shadow
// root so the host page's CSS doesn't bleed in.
//
// Flow:
//   1. Hover any element → blue outline + image count badge.
//   2. Click → smartLift to nearest sibling-rich ancestor → green
//      outline ("Đã chọn N ảnh"). User can press +/- to grow/shrink
//      the selection (parent / first descendant) or Enter to confirm,
//      Esc to cancel, click another element to re-select.
//   3. Confirm → primeLazyLoad + scrollPrimeAll → extract → resolve.

import { extractImages, primeLazyLoad, scrollPrimeAll } from '@core/sources/extract'
import { getSelectorPath } from '@core/selectors/path'
import { smartLift } from '@core/selectors/lift'

export interface PickerResult {
  selector:      string
  expectedCount: number
  images:        ReturnType<typeof extractImages>
}

export interface PickerOptions {
  hint?:        string
  /** When true, the auto-scroll toggle starts ON; the user can still
   *  flip it off in the toolbar. Default false — fast pick. */
  defaultAutoScroll?: boolean
  onConfirm:    (r: PickerResult) => void
  onCancel:     () => void
}

const HOST_ID = 'typoon-picker-host'

export function mountPicker(opts: PickerOptions): () => void {
  let host = document.getElementById(HOST_ID)
  if (host) host.remove()
  host = document.createElement('div')
  host.id = HOST_ID
  host.style.cssText =
    'all: initial; position: fixed; inset: 0; z-index: 2147483646; pointer-events: none;'
  document.documentElement.appendChild(host)

  const shadow = host.attachShadow({ mode: 'open' })
  shadow.innerHTML = TEMPLATE

  const hoverBox  = shadow.getElementById('hover-box') as HTMLDivElement
  const hoverBadge = shadow.getElementById('hover-badge') as HTMLDivElement
  const lockBox   = shadow.getElementById('lock-box')  as HTMLDivElement
  const lockBadge = shadow.getElementById('lock-badge') as HTMLDivElement
  const countBadge = shadow.getElementById('count-badge') as HTMLSpanElement
  const autoStatus = shadow.getElementById('autoscroll-status') as HTMLSpanElement
  const btnCancel = shadow.getElementById('btn-cancel')  as HTMLButtonElement
  const btnConfirm = shadow.getElementById('btn-confirm') as HTMLButtonElement
  const btnUp     = shadow.getElementById('btn-up')     as HTMLButtonElement
  const btnDown   = shadow.getElementById('btn-down')   as HTMLButtonElement

  let hovered: Element | null = null
  let locked:  Element | null = null
  let raf = 0
  let autoScroll = opts.defaultAutoScroll ?? false
  refreshAutoScrollStatus()

  function refreshAutoScrollStatus() {
    autoStatus.dataset.on = autoScroll ? '1' : '0'
  }

  function paintBox(box: HTMLDivElement, badge: HTMLDivElement, el: Element, label?: string) {
    cancelAnimationFrame(raf)
    raf = requestAnimationFrame(() => {
      const r = el.getBoundingClientRect()
      box.style.transform = `translate(${r.left}px, ${r.top}px)`
      box.style.width  = `${r.width}px`
      box.style.height = `${r.height}px`
      box.style.display = 'block'
      const count = quickImageCount(el)
      badge.textContent = label ?? (count > 0 ? `${count} ảnh` : 'không có ảnh')
      badge.dataset.empty = count === 0 ? '1' : '0'
    })
  }

  function clearHover() { hoverBox.style.display = 'none' }
  function clearLock()  { lockBox.style.display  = 'none' }

  function setCount(n: number, label?: string) {
    countBadge.textContent = label ?? `${n} ảnh`
    countBadge.dataset.empty = n === 0 ? '1' : '0'
  }

  function lockTo(el: Element, smart = true) {
    const target = smart ? smartLift(el) : el
    locked = target
    paintBox(lockBox, lockBadge, target, `Đã chọn · ${quickImageCount(target)} ảnh`)
    setCount(quickImageCount(target))
    btnConfirm.style.display = 'inline-flex'
    btnUp.style.display = 'inline-flex'
    btnDown.style.display = 'inline-flex'
  }

  function expand() {
    if (!locked) return
    const parent = locked.parentElement
    if (!parent || parent === document.documentElement) return
    locked = parent
    paintBox(lockBox, lockBadge, locked, `Đã chọn · ${quickImageCount(locked)} ảnh`)
    setCount(quickImageCount(locked))
  }

  function shrink() {
    if (!locked) return
    let best: Element | null = null
    let bestCount = -1
    for (const child of Array.from(locked.children)) {
      const c = quickImageCount(child)
      if (c > bestCount) { bestCount = c; best = child }
    }
    if (best && bestCount > 0) {
      locked = best
      paintBox(lockBox, lockBadge, locked, `Đã chọn · ${quickImageCount(locked)} ảnh`)
      setCount(quickImageCount(locked))
    }
  }

  function onMove(e: MouseEvent) {
    if (e.target === host) return
    const el = document.elementFromPoint(e.clientX, e.clientY)
    if (!el || el === host) return

    // Hover preview is always live, even when something is locked —
    // user needs to see what they'd switch to before clicking.
    const candidate = smartLift(el)
    if (!candidate || candidate === locked) { clearHover(); return }
    if (candidate === hovered) return
    hovered = candidate
    paintBox(hoverBox, hoverBadge, candidate)
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault()
      cleanup()
      opts.onCancel()
    } else if (e.key === 'Enter' && locked) {
      e.preventDefault()
      void confirm()
    } else if ((e.key === '+' || e.key === '=' || e.key === 'ArrowUp') && locked) {
      e.preventDefault()
      expand()
    } else if ((e.key === '-' || e.key === 'ArrowDown') && locked) {
      e.preventDefault()
      shrink()
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault()
      autoScroll = !autoScroll
      refreshAutoScrollStatus()
    }
  }

  function onClick(e: MouseEvent) {
    if (e.target === host) return
    e.preventDefault()
    e.stopPropagation()
    const el = document.elementFromPoint(e.clientX, e.clientY)
    if (!el || el === host) return
    clearHover()
    lockTo(el)
  }

  async function confirm() {
    if (!locked) return
    const target = locked
    cleanup()
    await primeLazyLoad(target)
    if (autoScroll) await scrollPrimeAll(target)
    const images = extractImages(target)
    opts.onConfirm({
      selector:      getSelectorPath(target),
      expectedCount: images.length,
      images,
    })
  }

  function cleanup() {
    document.removeEventListener('mousemove', onMove, true)
    document.removeEventListener('click', onClick, true)
    document.removeEventListener('keydown', onKey, true)
    cancelAnimationFrame(raf)
    host?.remove()
  }

  document.addEventListener('mousemove', onMove, true)
  document.addEventListener('click', onClick, true)
  document.addEventListener('keydown', onKey, true)
  btnCancel.addEventListener('click', () => { cleanup(); opts.onCancel() })
  btnConfirm.addEventListener('click', () => void confirm())
  btnUp.addEventListener('click',   expand)
  btnDown.addEventListener('click', shrink)

  if (opts.hint) {
    countBadge.title = `Đã từng dùng: ${opts.hint}`
  }

  // ── Auto-lock the most likely candidate on mount ─────────────────
  //
  // Find the largest visible region with multiple <img> children and
  // pre-lock it. The user sees a green outline + Confirm button
  // immediately on the page — no need to hunt for the right element.
  // They can override by clicking somewhere else.
  setTimeout(() => {
    const candidate = findBestCandidate()
    if (!candidate) return
    locked = candidate
    paintBox(lockBox, lockBadge, candidate, `Đã đoán · ${quickImageCount(candidate)} ảnh`)
    setCount(quickImageCount(candidate))
    btnConfirm.style.display = 'inline-flex'
    btnUp.style.display = 'inline-flex'
    btnDown.style.display = 'inline-flex'
  }, 50)

  return cleanup
}

/** Walk the DOM looking for the smallest container that holds the
 *  largest cluster of similarly-sized <img> elements. Used as a
 *  pre-selection on picker mount.
 *
 *  Heuristic:
 *   - Candidates: every element with ≥2 large-enough <img> children.
 *   - Score: image count × median image area.
 *   - Tie-break: prefer the smallest element (most specific).
 */
function findBestCandidate(): Element | null {
  const all = document.querySelectorAll<HTMLElement>('*')
  let best: Element | null = null
  let bestScore = 0

  for (const el of all) {
    const imgs = el.querySelectorAll<HTMLImageElement>(':scope > img, :scope > * > img')
    if (imgs.length < 2) continue

    const areas: number[] = []
    for (const img of imgs) {
      const r = img.getBoundingClientRect()
      if (r.width < 64 || r.height < 64) continue
      areas.push(r.width * r.height)
    }
    if (areas.length < 2) continue

    areas.sort((a, b) => a - b)
    const median = areas[Math.floor(areas.length / 2)]!
    const score = areas.length * median

    if (score > bestScore) {
      bestScore = score
      best = el
    }
  }
  return best
}

function quickImageCount(el: Element): number {
  let n = 0
  for (const img of Array.from(el.querySelectorAll<HTMLImageElement>('img'))) {
    const r = img.getBoundingClientRect()
    if (r.width >= 32 && r.height >= 32) n++
  }
  return n
}

const TEMPLATE = /* html */ `
  <style>
    :host { all: initial; }
    .box {
      position: fixed; inset: 0;
      width: 0; height: 0;
      display: none;
      pointer-events: none;
      border-radius: 4px;
      transition: transform 60ms linear, width 60ms linear, height 60ms linear;
    }
    #hover-box { box-shadow: 0 0 0 2px #4F88E6, 0 0 0 4px rgba(79, 136, 230, .25); }
    #lock-box  { box-shadow: 0 0 0 2px #FFA08C, 0 0 0 4px rgba(255, 160, 140, .3); }
    .badge {
      position: absolute; top: -22px; left: 0;
      background: #1A1410;
      font: 500 11px/1 system-ui, sans-serif;
      padding: 4px 6px; border-radius: 4px;
      white-space: nowrap;
    }
    #hover-badge { color: #8AB4F2; }
    #lock-badge  { color: #FFA08C; }
    .badge[data-empty="1"] { color: #80848E; }
    #toolbar {
      position: fixed; bottom: 16px; left: 50%;
      transform: translateX(-50%);
      pointer-events: auto;
      background: #1E1F22;
      color: #F2F3F5;
      font: 500 13px/1 system-ui, sans-serif;
      padding: 6px;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.06);
      display: flex; align-items: center; gap: 4px;
      max-width: calc(100vw - 32px);
      white-space: nowrap;
    }
    #count-badge {
      padding: 4px 10px;
      background: #2B2D31;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #FFA08C;
      margin-right: 4px;
    }
    #count-badge[data-empty="1"] { color: #80848E; }
    .sep {
      width: 1px; height: 18px;
      background: rgba(255,255,255,.08);
      flex: none;
      margin: 0 4px;
    }
    .btn {
      all: unset; cursor: pointer;
      padding: 6px 10px; border-radius: 6px;
      background: transparent; color: #B5BAC1;
      font: 500 12px/1 system-ui, sans-serif;
      display: none;
      align-items: center; justify-content: center;
      transition: background .12s, color .12s;
    }
    .btn:hover { color: #F2F3F5; background: #2B2D31; }
    .btn.icon { width: 28px; padding: 0; height: 28px; font-size: 14px; }
    .btn.primary { background: #FFA08C; color: #1A1410; display: inline-flex; }
    .btn.primary:hover { filter: brightness(1.08); }
    #autoscroll-status {
      display: none;
      padding: 4px 8px;
      background: rgba(79, 136, 230, .18);
      color: #8AB4F2;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 500;
    }
    #autoscroll-status[data-on="1"] { display: inline-flex; }
  </style>
  <div id="hover-box" class="box"><div id="hover-badge" class="badge">0 ảnh</div></div>
  <div id="lock-box" class="box"><div id="lock-badge" class="badge">Đã chọn</div></div>
  <div id="toolbar">
    <span id="count-badge" data-empty="1">0 ảnh</span>
    <span id="autoscroll-status" data-on="0" title="Auto-scroll bật (phím S)">↻ scroll</span>
    <button id="btn-down"  type="button" class="btn icon" title="Hẹp hơn (phím −)">−</button>
    <button id="btn-up"    type="button" class="btn icon" title="Rộng hơn (phím +)">+</button>
    <span class="sep"></span>
    <button id="btn-confirm" type="button" class="btn primary" title="Xác nhận (Enter)">Xác nhận</button>
    <button id="btn-cancel"  type="button" class="btn" style="display:inline-flex" title="Hủy (Esc)">Hủy</button>
  </div>
`
