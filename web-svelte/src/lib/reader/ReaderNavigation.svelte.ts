// reader/ReaderNavigation.svelte.ts — scroll reader tracking and chrome visibility.

export class ReaderNavigation {
  pageIndex = $state(0);
  scrollProgress = $state(0);
  chromeVisible = $state(true);

  #pageCount: () => number;
  #stripScrollFrame = 0;
  #pendingStripScrollEl: HTMLDivElement | null = null;

  constructor(pageCount: () => number) {
    this.#pageCount = pageCount;

    $effect(() => {
      const max = Math.max(0, this.#pageCount() - 1);
      if (this.pageIndex > max) this.pageIndex = max;
    });
  }

  toggleChrome(): void {
    this.chromeVisible = !this.chromeVisible;
  }

  // ── strip scroll tracking ──

  handleStripScroll = (event: Event): void => {
    this.#pendingStripScrollEl = event.currentTarget as HTMLDivElement;
    if (this.#stripScrollFrame) return;
    this.#stripScrollFrame = requestAnimationFrame(() => {
      this.#stripScrollFrame = 0;
      const el = this.#pendingStripScrollEl;
      this.#pendingStripScrollEl = null;
      if (el) this.#updateStripScroll(el);
    });
  };

  #updateStripScroll(el: HTMLDivElement): void {
    const center = el.scrollTop + el.clientHeight / 2;
    const max = el.scrollHeight - el.clientHeight;
    this.scrollProgress = max > 0 ? Math.min(1, Math.max(0, el.scrollTop / max)) : 0;

    const best = nearestPageIndex(el, center);
    if (best !== this.pageIndex) this.pageIndex = best;
  }

  // ── tap handlers ──

  handleStripTap = (event: MouseEvent): void => {
    if (isInteractive(event.target)) return;
    const target = event.currentTarget as HTMLDivElement;
    const rect = target.getBoundingClientRect();
    const xPct = (event.clientX - rect.left) / rect.width;
    if (xPct > 0.33 && xPct < 0.67) this.toggleChrome();
  };
}

function isInteractive(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && !!target.closest('button,a,input,textarea,select,[role="button"]');
}

// Page elements stack top→bottom in DOM order, so their vertical midpoints are
// monotonically increasing. Binary-search for the page nearest the viewport
// center instead of scanning every node each scroll frame.
function nearestPageIndex(el: HTMLDivElement, center: number): number {
  const nodes = el.querySelectorAll<HTMLElement>('[data-page-index]');
  if (!nodes.length) return 0;

  // Page midpoint in the strip's scroll-content space. Don't use offsetTop here:
  // each page sits in its own positioned wrapper, so offsetTop is measured
  // against that wrapper (≈0) instead of accumulating down the strip — which
  // collapses every page's midpoint to ~pageHeight/2 and makes the binary search
  // jump straight to the last page. getBoundingClientRect is positioning-agnostic.
  const stripTop = el.getBoundingClientRect().top;
  const midOf = (node: HTMLElement): number => {
    const rect = node.getBoundingClientRect();
    return rect.top - stripTop + el.scrollTop + rect.height / 2;
  };

  let lo = 0;
  let hi = nodes.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (midOf(nodes[mid]!) < center) lo = mid + 1;
    else hi = mid;
  }

  // lo is the first node at/after center; the nearest is lo or lo-1.
  let best = nodes[lo]!;
  if (lo > 0 && Math.abs(midOf(nodes[lo - 1]!) - center) <= Math.abs(midOf(best) - center)) {
    best = nodes[lo - 1]!;
  }
  return Number(best.dataset.pageIndex) || 0;
}
