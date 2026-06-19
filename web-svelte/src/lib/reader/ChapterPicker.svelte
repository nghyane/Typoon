<script lang="ts">
  import { Search } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import ReaderDropdown from './ReaderDropdown.svelte';
  import type { ReaderChapterLink } from '$lib/types';

  let {
    open,
    anchor,
    onClose,
    chapters,
    workId,
    currentRef,
  }: {
    open: boolean;
    anchor: HTMLElement | null;
    onClose: () => void;
    chapters: readonly ReaderChapterLink[];
    workId: string;
    currentRef: string;
  } = $props();

  let q = $state('');
  const term = $derived(q.trim().toLowerCase());
  const rows = $derived(term
    ? chapters.filter((chapter) => `${chapter.number} ${chapter.label}`.toLowerCase().includes(term))
    : chapters);

  function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function chapterSubtitle(number: string, label: string | null | undefined): string | null {
    if (!label) return null;
    let rest = label.trim();
    if (!rest) return null;
    const patterns = [
      new RegExp(`^(?:ch(?:apter|ương|uong)?)\\s*\\.?\\s*${escapeRegExp(number)}\\s*[:.\\-–—]?\\s*`, 'i'),
      /^(?:ch(?:apter|ương|uong)?)\s*\.?\s*\d+(?:\.\d+)?\s*[:.\-–—]?\s*/i,
    ];
    for (const pattern of patterns) {
      const next = rest.replace(pattern, '');
      if (next !== rest) {
        rest = next;
        break;
      }
    }
    rest = rest.trim();
    if (!rest || rest === number) return null;
    return rest;
  }
</script>

<ReaderDropdown {open} {anchor} {onClose} align="end" width="min(26rem, calc(100vw - 1rem))">
  <div class="flex flex-col max-h-[min(70dvh,34rem)]">
    <div class="flex items-center justify-between gap-3 px-3 py-2 border-b border-divider">
      <div class="min-w-0">
        <div class="text-sm font-semibold text-text">Danh sách chương</div>
        <div class="text-xs text-text-subtle">{chapters.length} chương</div>
      </div>
    </div>
    <div class="px-3 py-2 bg-surface z-10">
      <div class="relative">
        <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
        <input
          type="search"
          bind:value={q}
          placeholder="Tìm chương…"
          class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors"
        />
      </div>
    </div>

    <div class="flex-1 overflow-y-auto p-2">
      {#if rows.length === 0}
        <div class="px-3 py-8 text-center text-sm text-text-subtle">Không khớp tìm kiếm</div>
      {:else}
        <div class="space-y-1">
          {#each rows as chapter (chapter.numberNorm)}
            {@const active = chapter.numberNorm === currentRef}
            {@const subtitle = chapterSubtitle(chapter.number, chapter.label)}
            <a
              href={`/r/${workId}/${chapter.numberNorm}`}
              onclick={onClose}
              class={cn(
                'relative flex items-center gap-3 min-h-9 px-3 rounded-sm text-sm transition-colors duration-150 cursor-pointer',
                active ? 'bg-accent-bg text-accent-text font-medium' : 'text-text-muted hover:text-text hover:bg-hover',
              )}
            >
              {#if active}<span aria-hidden="true" class="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"></span>{/if}
              <span class="tabular-nums font-medium shrink-0">Ch.{chapter.number || chapter.numberNorm}</span>
              {#if subtitle}<span class={cn('truncate text-xs', active ? 'text-accent-text' : 'text-text-subtle')}>{subtitle}</span>{/if}
            </a>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</ReaderDropdown>
