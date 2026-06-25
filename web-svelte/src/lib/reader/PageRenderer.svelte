<script lang="ts">
  import { Loader2 } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import type { PageBlob } from '$lib/types';

  let {
    blob,
    index,
    pageSize,
    eager = false,
    className = '',
    register,
  }: {
    blob: PageBlob;
    index: number;
    pageSize?: { width: number; height: number } | null;
    eager?: boolean;
    className?: string;
    /** Registers this frame as the overlay surface for `index`. Returns cleanup. */
    register?: (el: HTMLElement) => (() => void) | void;
  } = $props();

  const LAZY_RENDER_MARGIN = '1800px 0px';

  let host = $state<HTMLDivElement | null>(null);
  let objectUrl = $state('');
  let ready = $state(false);
  let failed = $state(false);
  let nearViewport = $state(false);
  let w = $state(0);
  let h = $state(0);

  // The frame is aspect-locked to the source page so the overlay (% of this box)
  // coincides with the displayed image regardless of when bytes load. Prefer the
  // host-provided size (single source of truth), fall back to the decoded image,
  // then to a neutral placeholder until either is known.
  const aspect = $derived(
    pageSize?.width && pageSize.height ? `${pageSize.width} / ${pageSize.height}`
    : w && h ? `${w} / ${h}`
    : '2 / 3',
  );
  const shouldRender = $derived(!!blob && (eager || nearViewport));

  // container-type:inline-size makes 1cqw === 1% of this frame's width, which is
  // the page's displayed width — the unit the overlay's typography scales by.
  const frameStyle = $derived(`aspect-ratio:${aspect};container-type:inline-size`);

  $effect(() => {
    const el = host;
    if (!el) return;
    if (eager || typeof IntersectionObserver === 'undefined') { nearViewport = true; return; }
    const observer = new IntersectionObserver(entries => {
      nearViewport = entries.some(entry => entry.isIntersecting);
    }, { rootMargin: LAZY_RENDER_MARGIN });
    observer.observe(el);
    return () => observer.disconnect();
  });

  $effect(() => {
    const src = shouldRender ? blob : null;
    if (!src) { ready = false; failed = false; w = 0; h = 0; objectUrl = ''; return; }
    const url = URL.createObjectURL(src);
    objectUrl = url; ready = false; failed = false;
    return () => { URL.revokeObjectURL(url); if (objectUrl === url) objectUrl = ''; };
  });

  function registerFrame(node: HTMLElement) {
    const cleanup = register?.(node);
    return { destroy() { cleanup?.(); } };
  }
</script>

<div
  bind:this={host}
  use:registerFrame
  class={cn('relative w-full overflow-hidden bg-bg', className)}
  style={frameStyle}
  data-index={index}
  data-page-index={index}
>
  {#if objectUrl && !failed}
    <img
      src={objectUrl}
      loading={eager ? 'eager' : 'lazy'}
      decoding="async"
      class="absolute inset-0 w-full h-full object-contain select-none"
      style={`opacity:${ready ? 1 : 0}`}
      alt={`Trang ${index + 1}`}
      onload={(event) => { const img = event.currentTarget as HTMLImageElement; w = img.naturalWidth; h = img.naturalHeight; ready = true; }}
      onerror={() => { failed = true; }}
    />
  {/if}
  {#if !blob || !ready || failed}
    <div class="absolute inset-0 grid place-items-center bg-surface-2/35 text-text-subtle">
      {#if failed}<div class="text-xs">Không đọc được trang {index + 1}</div>
      {:else}<div class="inline-flex items-center gap-2 text-xs"><Loader2 size={14} class="animate-spin" /> Trang {index + 1}</div>{/if}
    </div>
  {/if}
</div>
