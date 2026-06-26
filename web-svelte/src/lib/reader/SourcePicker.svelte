<script lang="ts">
  import { cn } from '$lib/cn';
  import { trackSourceSelect } from '$lib/analytics/client';
  import ReaderDropdown from './ReaderDropdown.svelte';
  import type { ReaderSourceVersion } from '$lib/types';

  let {
    open,
    anchor,
    onClose,
    versions,
    activeKey,
    targetLang,
    onPick,
    busy = false,
  }: {
    open: boolean;
    anchor: HTMLElement | null;
    onClose: () => void;
    versions: readonly ReaderSourceVersion[];
    activeKey: string | null;
    targetLang: string;
    onPick: (version: ReaderSourceVersion) => void;
    busy?: boolean;
  } = $props();

  const sortedVersions = $derived([...versions].sort((a, b) => {
    const target = targetLang.toLowerCase();
    if (a.lang === target && b.lang !== target) return -1;
    if (a.lang !== target && b.lang === target) return 1;
    return (b.date ?? '').localeCompare(a.date ?? '');
  }));

  function timeAgo(value: string): string {
    const time = Date.parse(value);
    if (!Number.isFinite(time)) return value.slice(0, 10);
    const diff = Math.max(0, Date.now() - time);
    const minutes = Math.floor(diff / 60_000);
    if (minutes < 1) return 'vừa xong';
    if (minutes < 60) return `${minutes} phút trước`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} giờ trước`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days} ngày trước`;
    return value.slice(0, 10);
  }

  function pick(version: ReaderSourceVersion): void {
    trackSourceSelect({
      context: 'reader',
      source_id: version.sourceId,
      source_name: version.sourceName,
      lang: version.lang,
      active: activeKey === version.key,
    });
    onPick(version);
    onClose();
  }
</script>

<ReaderDropdown {open} {anchor} {onClose} align="end" width="min(18rem, calc(100vw - 1rem))" widthPx={288}>
  <div class="flex flex-col max-h-[min(60dvh,22rem)] py-1.5">
    {#if sortedVersions.length === 0}
      <p class="px-4 py-6 text-sm text-text-subtle text-center">Chưa có nguồn.</p>
    {:else}
      <div class="flex-1 overflow-y-auto px-1.5">
        {#each sortedVersions as version (version.key)}
          {@const active = activeKey === version.key}
          <button
            type="button"
            onclick={() => { pick(version); }}
            aria-pressed={active}
            disabled={busy}
            class={cn(
              'relative grid w-full grid-cols-[1.625rem_minmax(0,1fr)] items-center gap-1.5 min-h-8 px-2 rounded-sm text-left transition-colors duration-150 cursor-pointer disabled:opacity-60 disabled:cursor-wait',
              active ? 'bg-row-active text-text' : 'text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {#if active}<span aria-hidden="true" class="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"></span>{/if}
            <span class={cn('inline-flex h-5 items-center justify-center rounded-xs bg-surface-2 px-1 text-[10px] uppercase font-semibold tabular-nums', active ? 'text-accent' : 'text-text-subtle')}>{version.lang ? version.lang.toUpperCase() : '?'}</span>
            <span class="min-w-0 leading-tight">
              <span class={cn('block truncate text-xs font-medium', active ? 'text-text' : 'text-text-muted')}>{version.sourceName}</span>
              {#if version.scanlator}
                <span class="block truncate text-[11px] text-text-subtle">@{version.scanlator}</span>
              {:else if version.date}
                <span class="block truncate text-[11px] text-text-subtle">{timeAgo(version.date)}</span>
              {/if}
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</ReaderDropdown>
