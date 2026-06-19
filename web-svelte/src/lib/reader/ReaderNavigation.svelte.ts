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

    let best = 0;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const node of el.querySelectorAll<HTMLElement>('[data-page-index]')) {
      const mid = node.offsetTop + node.offsetHeight / 2;
      const dist = Math.abs(mid - center);
      if (dist < bestDist) {
        bestDist = dist;
        best = Number(node.dataset.pageIndex) || 0;
      }
    }
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
