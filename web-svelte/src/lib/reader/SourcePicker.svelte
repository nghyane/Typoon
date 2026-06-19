<script lang="ts">
  import { cn } from '$lib/cn';
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
</script>

<ReaderDropdown {open} {anchor} {onClose} align="end" width="min(24rem, calc(100vw - 1rem))">
  <div class="flex flex-col max-h-[min(70dvh,30rem)]">
    <div class="px-3 py-2 border-b border-divider">
      <div class="text-sm font-semibold text-text">Nguồn đọc</div>
      <div class="text-xs text-text-subtle">Ưu tiên ngôn ngữ đích {targetLang.toUpperCase()}</div>
    </div>
    {#if sortedVersions.length === 0}
      <p class="px-4 py-6 text-sm text-text-subtle text-center">Chưa có nguồn.</p>
    {:else}
      <div class="flex-1 overflow-y-auto p-2">
        {#each sortedVersions as version (version.key)}
          {@const active = activeKey === version.key}
          <button
            type="button"
            onclick={() => { onPick(version); onClose(); }}
            aria-pressed={active}
            disabled={busy}
            class={cn(
              'grid w-full grid-cols-[1rem_2rem_minmax(0,1fr)] items-center gap-2 h-8 px-2 rounded-sm text-left transition-colors duration-150 cursor-pointer disabled:opacity-60 disabled:cursor-wait',
              active ? 'text-text' : 'hover:bg-hover',
            )}
          >
            <span class="text-accent text-sm leading-none">{active ? '✓' : ''}</span>
            <span class={cn('text-xs uppercase font-semibold tabular-nums', active ? 'text-accent' : 'text-text-subtle')}>{version.lang.toUpperCase()}</span>
            <span class="min-w-0 truncate text-sm text-text">
              {version.sourceName}
              {#if version.scanlator}
                <span class="text-text-subtle"> · @{version.scanlator}</span>
              {:else if version.date}
                <span class="text-text-subtle"> · {timeAgo(version.date)}</span>
              {/if}
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</ReaderDropdown>
