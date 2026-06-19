<script lang="ts">
  import { Loader2 } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import type { PageBlob } from '$lib/types';

  let { blob, index, pageSize, eager = false, className = '' }: { blob: PageBlob; index: number; pageSize?: { width: number; height: number } | null; eager?: boolean; className?: string; } = $props();

  const LAZY_RENDER_MARGIN = '1800px 0px';

  let host = $state<HTMLDivElement | null>(null);
  let objectUrl = $state('');
  let ready = $state(false);
  let failed = $state(false);
  let nearViewport = $state(false);
  let w = $state(0);
  let h = $state(0);
  const ratio = $derived(pageSize?.width && pageSize.height ? pageSize.height / pageSize.width : w && h ? h / w : 1.5);
  const shouldRender = $derived(!!blob && (eager || nearViewport));

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
</script>

<div bind:this={host} class={cn('relative w-full bg-bg', className)} style={`padding-top:${ratio * 100}%`} data-index={index}>
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
