<script lang="ts">
  import { cn } from '$lib/cn';
  import { useSourceFetch } from '$lib/sourceFetch.svelte';

  let {
    src = null,
    title,
    alt,
    class: cls = '',
    fontSize = 'text-xl',
    version,
    aspect,
  }: {
    src?: string | null;
    title?: string | null;
    alt?: string | null;
    class?: string;
    fontSize?: string;
    version?: string | null;
    aspect?: string;
  } = $props();

  const sf = useSourceFetch();
  let failed = $state(false);
  const safeTitle = $derived((title ?? alt ?? '').trim());
  const url = $derived(!failed ? coverUrl(src, version, sf.toBrowserUrl) : null);

  $effect(() => {
    src;
    failed = false;
  });

  function coverUrl(
    value: string | null | undefined,
    v: string | null | undefined,
    proxify: (url: string) => string,
  ): string | null {
    if (!value) return null;
    const isAbs = /^https?:\/\//i.test(value);
    const base = isAbs ? proxify(value) : value;
    if (!v) return base;
    return base.includes('?')
      ? `${base}&v=${encodeURIComponent(v)}`
      : `${base}?v=${encodeURIComponent(v)}`;
  }
</script>

<div class={cn('flex items-center justify-center overflow-hidden bg-surface-2', aspect, cls)}>
  {#if url}
    <img src={url} alt={safeTitle} class="w-full h-full object-cover" loading="lazy" onerror={() => { failed = true; }} />
  {:else}
    <span class={cn('font-black text-text-subtle/60 select-none', fontSize)}>
      {safeTitle.slice(0, 2).toUpperCase() || '—'}
    </span>
  {/if}
</div>
