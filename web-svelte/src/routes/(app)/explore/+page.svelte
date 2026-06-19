<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { Check, Compass, Plus, Search } from 'lucide-svelte';
  import { listEnabledSources } from '$lib/source/registry';
  import {
    assembleFilterParams,
    assembleFilterState,
    getDefaultFilterState,
    getShelves,
    hasSearch,
    searchPageSize,
    shelfPageSize,
  } from '$lib/source/runtime/metadata';
  import { fetchBrowse } from '$lib/source/runtime/endpoints';
  import type { InstalledSource, MangaSummary } from '$lib/source/types';
  import { listWorksBySourceRefs, ensureWorkFromSource } from '$lib/works/repo';
  import type { Work } from '$lib/db';
  import Cover from '$lib/ui/Cover.svelte';
  import AddMangaModal from '$lib/library/AddMangaModal.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import { cn } from '$lib/cn';
  import { createInfiniteQuery, createMutation } from '@tanstack/svelte-query';

  let sources = $state<InstalledSource[]>([]);
  let sourceId = $state('');
  let shelfId = $state('');
  let query = $state('');
  let debouncedQuery = $state('');
  let addOpen = $state(false);
  let pendingMangaKey = $state<string | null>(null);
  let openError = $state('');

  let existingByUrl = $state<Record<string, Pick<Work, 'id' | 'in_library'>>>({});
  let loadMoreSentinel = $state<HTMLDivElement | null>(null);

  const source = $derived(sources.find((s) => s.manifest.id === sourceId) ?? null);
  const shelves = $derived(source ? getShelves(source.manifest) : []);
  const canSearch = $derived(source ? hasSearch(source.manifest) : false);
  const filters = $derived(source ? getDefaultFilterState(source.manifest) : {});
  const filterParams = $derived(source ? assembleFilterParams(source.manifest, filters) : '');
  const typedFilterState = $derived(source ? assembleFilterState(source.manifest, filters) : {});

  const target = $derived(
    canSearch && debouncedQuery ? { search: true as const } : shelfId || null,
  );

  const browsePageSize = $derived(
    source && target
      ? (typeof target === 'object' ? searchPageSize(source.manifest) : shelfPageSize(source.manifest, target))
      : Infinity,
  );
  const paginated = $derived(browsePageSize !== Infinity);

  // ── Infinite query (replaces manual loading/pagination/$effect) ──

  const browseQuery = createInfiniteQuery(() => ({
    queryKey: ['explore', source?.manifest.id, target, debouncedQuery, filterParams] as const,
    queryFn: ({ pageParam }: { pageParam: number }) => {
      const src = source!;
      const t = target!;
      return fetchBrowse(src.manifest, t, {
        page: pageParam,
        q: debouncedQuery || undefined,
        filterParams,
        filterState: typedFilterState,
      });
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage, _allPages, lastPageParam) => {
      if (!paginated) return undefined;
      if (lastPage.length < browsePageSize) return undefined;
      return (lastPageParam as number) + 1;
    },
    enabled: !!source && !!target,
  }));

  const items = $derived(browseQuery.data?.pages.flat() ?? []);

  // ── Library badge lookup ──────────────────────────────────────

  $effect(() => {
    if (!source || items.length === 0) {
      existingByUrl = {};
      return;
    }
    let cancelled = false;
    listWorksBySourceRefs(source.manifest.id, items.map((it) => it.url))
      .then((works) => {
        if (cancelled) return;
        existingByUrl = Object.fromEntries(
          [...works.entries()].map(([url, w]) => [url, { id: w.id, in_library: w.in_library }]),
        );
      });
    return () => { cancelled = true; };
  });

  // ── Infinite sentinel ─────────────────────────────────────────

  $effect(() => {
    const node = loadMoreSentinel;
    if (!node || !browseQuery.hasNextPage || browseQuery.isFetchingNextPage) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) browseQuery.fetchNextPage();
    }, { rootMargin: '800px 0px' });
    observer.observe(node);
    return () => observer.disconnect();
  });

  // ── Source / shelf init ───────────────────────────────────────

  onMount(() => {
    sources = listEnabledSources();
    sourceId = sources[0]?.manifest.id ?? '';
  });

  $effect(() => {
    if (!source) return;
    shelfId = shelves[0]?.id ?? '';
    query = '';
    debouncedQuery = '';
  });

  $effect(() => {
    const v = query;
    const timer = window.setTimeout(() => { debouncedQuery = v.trim(); }, 400);
    return () => window.clearTimeout(timer);
  });

  // ── Card navigation ───────────────────────────────────────────

  const ensureMutation = createMutation(() => ({
    mutationFn: (manga: MangaSummary) =>
      ensureWorkFromSource($state.snapshot(source!.manifest), manga),
  }));

  function mangaKey(manga: MangaSummary): string {
    return `${source?.manifest.id ?? ''}::${manga.url}`;
  }

  async function openManga(manga: MangaSummary): Promise<void> {
    if (!source || pendingMangaKey) return;
    const key = mangaKey(manga);
    pendingMangaKey = key;
    openError = '';
    try {
      const work = await ensureMutation.mutateAsync(manga);
      await goto(`/w/${work.id}`);
    } catch (err) {
      openError = err instanceof Error ? err.message : String(err);
    } finally {
      if (pendingMangaKey === key) pendingMangaKey = null;
    }
  }
</script>

<svelte:head><title>Khám phá — Hội Mê Truyện</title></svelte:head>

<div class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">
  {#if sources.length === 0}
    <div class="max-w-3xl mx-auto px-4 sm:px-6 py-10">
      <EmptyState title="Chưa có nguồn nào" hint="Mở Settings → Nguồn để bật ít nhất một nguồn." />
    </div>
  {:else}
    <header class="space-y-3">
      <div class="flex items-center justify-between gap-3">
        <h1 class="text-lg font-semibold text-text inline-flex items-center gap-2">
          <Compass size={18} /> Khám phá
        </h1>
        <Button variant="secondary" size="sm" onclick={() => { addOpen = true; }}>
          <Plus size={14} /> Thêm
        </Button>
      </div>
      <div class="flex flex-wrap gap-2" role="tablist" aria-label="Nguồn truyện">
        {#each sources as s (s.manifest.id)}
          {@const selected = s.manifest.id === sourceId}
          <button type="button" role="tab" aria-selected={selected}
            onclick={() => { sourceId = s.manifest.id; }}
            class={cn('inline-flex items-center gap-2 h-7 px-3 rounded-full text-xs font-medium transition-colors',
              selected ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >{s.manifest.name}</button>
        {/each}
      </div>
      {#if source && canSearch}
        <div class="relative max-w-xl">
          <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
          <input type="search" bind:value={query}
            placeholder={`Tìm trong ${source.manifest.name}…`}
            class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors"
          />
        </div>
      {/if}
    </header>

    {#if !debouncedQuery && shelves.length > 1}
      <div class="flex flex-wrap gap-2">
        {#each shelves as shelf (shelf.id)}
          {@const selected = shelf.id === shelfId}
          <button type="button" onclick={() => { shelfId = shelf.id; }}
            class={cn('h-7 px-3 rounded-full text-xs font-medium transition-colors',
              selected ? 'bg-surface-2 text-text' : 'text-text-muted hover:text-text',
            )}
          >{shelf.label}</button>
        {/each}
      </div>
    {/if}

    {#if browseQuery.isPending}
      <div class="flex justify-center py-12"><Spinner size={20} /></div>
    {:else if browseQuery.error && items.length === 0}
      <div class="py-12 text-center">
        <p class="text-sm font-medium text-error-text">Không tải được</p>
        <p class="text-xs text-text-subtle mt-1">{browseQuery.error.message}</p>
      </div>
    {:else if items.length === 0}
      <div class="py-12 text-center">
        <p class="text-sm font-medium text-text">{debouncedQuery ? 'Không tìm thấy' : 'Chưa có truyện ở đây'}</p>
        {#if debouncedQuery}<p class="text-xs text-text-subtle mt-1">Thử từ khoá khác.</p>{/if}
      </div>
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
        {#each items as manga (manga.id)}
          {@const existing = existingByUrl[manga.url]}
          {@const key = mangaKey(manga)}
          {@const pending = pendingMangaKey === key}
          <button type="button" onclick={() => openManga(manga)}
            disabled={pendingMangaKey !== null}
            aria-busy={pending}
            class={cn(
              'group flex flex-col gap-2 rounded-sm overflow-hidden text-left cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed',
              pendingMangaKey !== null && !pending && 'opacity-50',
              pending && 'cursor-wait',
            )}
          >
            <div class="relative aspect-[3/4] rounded-sm overflow-hidden bg-surface-2">
              <Cover src={manga.cover} title={manga.title} class="absolute inset-0 transition-transform group-hover:scale-[1.02]" />
              {#if pending}
                <span class="absolute inset-0 grid place-items-center bg-bg/45"><Spinner size={20} /></span>
              {/if}
              {#if source?.manifest.nsfw}
                <span class="absolute top-1.5 right-1.5 inline-flex items-center h-5 px-1.5 rounded-xs bg-error-bg text-error-text text-xs font-semibold">18+</span>
              {/if}
            </div>
            <div class="px-0.5 space-y-0.5 min-w-0">
              <div class="text-xs font-medium text-text line-clamp-2 leading-tight">{manga.title}</div>
              <div class="flex items-center gap-1.5 text-xs uppercase tracking-wider text-text-subtle">
                <span class="truncate">{source?.manifest.name}</span>
                {#if existing?.in_library}
                  <Check size={12} class="shrink-0 text-success" aria-label="Trong thư viện" />
                {/if}
              </div>
            </div>
          </button>
        {/each}
      </div>

      {#if openError || ensureMutation.error}
        <div class="py-3 text-center"><p class="text-sm text-error-text">{openError || ensureMutation.error?.message}</p></div>
      {/if}

      {#if browseQuery.error && items.length > 0}
        <div class="py-2 text-center space-y-2">
          <p class="text-sm text-error-text">{browseQuery.error.message}</p>
          <button type="button" onclick={() => browseQuery.fetchNextPage()}
            class="inline-flex items-center justify-center h-8 px-3 rounded-sm bg-surface-2 text-text text-sm font-medium hover:bg-interactive-hover transition-colors cursor-pointer"
          >Thử lại</button>
        </div>
      {/if}

      {#if browseQuery.hasNextPage}
        <div bind:this={loadMoreSentinel} class="flex justify-center pt-3 min-h-12">
          {#if browseQuery.isFetchingNextPage}
            <div class="inline-flex items-center gap-2 text-sm text-text-subtle">
              <Spinner size={16} /> Đang tải thêm
            </div>
          {/if}
        </div>
      {/if}
    {/if}
  {/if}
  <AddMangaModal open={addOpen} onClose={() => { addOpen = false; }} />
</div>
