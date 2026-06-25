<script lang="ts">
  import { Plus, X } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import type { Work } from '$lib/db';
  import type { InstalledSource } from '$lib/source/types';
  import Cover from '$lib/ui/Cover.svelte';

  let {
    work,
    sourceMap = new Map<string, InstalledSource | null>(),
    detailLoading = false,
    detailFailures = 0,
    detachPending = false,
    detachError = '',
    onDetach = (_source: string, _upstreamRef: string) => {},
    onAttach = () => {},
  }: {
    work: Work;
    sourceMap: Map<string, InstalledSource | null>;
    detailLoading: boolean;
    detailFailures: number;
    detachPending?: boolean;
    detachError?: string;
    onDetach: (source: string, upstreamRef: string) => void;
    onAttach: () => void;
  } = $props();

  function sourceLabel(id: string): string {
    return sourceMap.get(id)?.manifest.name ?? id;
  }

  function sourceImageHeaders(id: string): Record<string, string> | undefined {
    return sourceMap.get(id)?.manifest.imageHeaders;
  }
</script>

<section class="pt-1 pb-3">
  <div class="flex items-center gap-2 mb-2">
    <h2 class="text-xs uppercase tracking-wider text-text-subtle font-medium">Nguồn</h2>
    <span class="text-xs text-text-subtle tabular-nums">{work.sources.length}</span>
    {#if detailLoading}<span class="ts-spinner-circle size-3 text-text-subtle" aria-label="Đang tải"></span>{/if}
    {#if detailFailures > 0}<span class="text-xs text-warning-text">{detailFailures} nguồn lỗi</span>{/if}
  </div>
  <div class="flex flex-wrap gap-2">
    {#each work.sources as source, i (`${source.source}:${source.upstream_ref}`)}
      <div class={cn(
        'flex-1 basis-[200px] min-w-[180px] max-w-[320px] flex items-center gap-2 h-11 pl-2 pr-1 rounded-sm bg-surface-2 group/src',
        i === 0 && work.sources.length > 1 && 'border-l-2 border-accent',
      )}>
        <div class="w-6 h-8 shrink-0 rounded-xs overflow-hidden">
          <Cover src={source.cover_url} headers={sourceImageHeaders(source.source)} title={source.title} class="w-full h-full" fontSize="text-[10px]" />
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-sm text-text truncate leading-tight">{source.title}</p>
          <p class="text-xs text-text-subtle truncate">
            {sourceLabel(source.source)}{#if i === 0 && work.sources.length > 1}<span class="ml-1 text-accent">Chính</span>{/if}
          </p>
        </div>
        {#if work.sources.length > 1}
          <button type="button"
            onclick={() => { if (confirm(`Gỡ nguồn "${sourceLabel(source.source)}"?`)) onDetach(source.source, source.upstream_ref); }}
            disabled={detachPending}
            class="shrink-0 size-6 rounded-sm flex items-center justify-center text-text-subtle hover:text-error-text hover:bg-error-bg opacity-0 group-hover/src:opacity-100 transition-opacity cursor-pointer disabled:opacity-40 disabled:cursor-wait"
            aria-label="Gỡ nguồn">
            <X size={12} />
          </button>
        {/if}
      </div>
    {/each}
    <button type="button" onclick={onAttach}
      class="flex-1 basis-[200px] min-w-[180px] max-w-[320px] flex items-center gap-2 h-11 px-2 rounded-sm text-sm text-text-muted bg-transparent hover:bg-surface-2 hover:text-text border border-dashed border-border-soft transition-colors cursor-pointer text-left">
      <span class="w-6 h-8 shrink-0 rounded-xs flex items-center justify-center bg-surface-2"><Plus size={12} class="text-text-subtle" /></span>
      Thêm nguồn
    </button>
  </div>
  {#if detachError}<p class="text-xs text-error-text mt-1">{detachError}</p>{/if}
</section>
