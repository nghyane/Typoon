<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { Compass, Plus, RotateCw, Search } from 'lucide-svelte';
  import { listSources } from '$lib/source/registry';
  import {
    assembleFilters,
    assembleFilterState,
    getDefaultFilterState,
    getFilters,
    getShelves,
    hasSearch,
    searchPageSize,
    shelfPageSize,
  } from '$lib/source/runtime/metadata';
  import { fetchBrowse } from '$lib/source/runtime/endpoints';
  import { dedupeBy } from '$lib/collections';
  import FilterChips from '$lib/source/FilterChips.svelte';
  import FilterGroup from '$lib/source/FilterGroup.svelte';
  import type { Filter, InstalledSource, MangaSummary } from '$lib/source/types';
  import { listWorksBySourceRefs, ensureWorkFromSource } from '$lib/works/repo';
  import type { Work } from '$lib/db';
  import WorkCard from '$lib/ui/WorkCard.svelte';
  import AddMangaModal from '$lib/library/AddMangaModal.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import CardSkeleton from '$lib/ui/CardSkeleton.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import { cn } from '$lib/cn';
  import { trackSourceOpen, trackSourceSelect } from '$lib/analytics/client';
  import { createInfiniteQuery, createMutation, keepPreviousData } from '@tanstack/svelte-query';

  let allSources = $state<InstalledSource[]>([]);
  let sourceId = $state('');
  let shelfId = $state('');
  let query = $state('');
  let debouncedQuery = $state('');
  let addOpen = $state(false);
  let pendingMangaKey = $state<string | null>(null);
  let openError = $state('');

  let existingByUrl = $state<Record<string, Pick<Work, 'id' | 'in_library'>>>({});
  let loadMoreSentinel = $state<HTMLDivElement | null>(null);
  let rootEl = $state<HTMLDivElement | null>(null);
  let scrollEl: HTMLElement | null = null;
  let pendingScrollTop = $state<number | null>(null);

  const sources = $derived(allSources.filter((s) => s.enabled));
  const disabledSourceCount = $derived(allSources.filter((s) => !s.enabled).length);
  const source = $derived(sources.find((s) => s.manifest.id === sourceId) ?? null);
  const shelves = $derived(source ? getShelves(source.manifest) : []);
  const canSearch = $derived(source ? hasSearch(source.manifest) : false);

  // Interactive filter selection (resets to source defaults when the source changes).
  let filterState = $state<Record<string, string | string[]>>({});

  // A query-inject filter (e.g. nhentai tags) only works through the search
  // endpoint, so an active one forces the search target even without typed text.
  const sourceFilters = $derived(source ? getFilters(source.manifest) : []);
  const queryInjectActive = $derived(
    sourceFilters.some((f) => f.inject === 'query' && filterState[f.id] != null),
  );

  const target = $derived(
    canSearch && (debouncedQuery || queryInjectActive) ? { search: true as const } : shelfId || null,
  );

  // Filters scoped to the active target: source-level for search, shelf-level
  // (falling back to source-level) for a shelf.
  const activeFilters = $derived(
    !source || target == null
      ? []
      : typeof target === 'object'
        ? getFilters(source.manifest)
        : getFilters(source.manifest, target),
  );
  const assembled = $derived(assembleFilters(activeFilters, filterState));
  const typedFilterState = $derived(assembleFilterState(activeFilters, filterState));

  // Root cause (all providers): a filter declared on a *shelf* (rather than
  // source-level) is only visible once that shelf is the active one — so
  // "browse by genre / tag / catalog" gets buried behind first selecting its
  // shelf (TruyenQQ/Naver/otruyen/webtoonscan/happymh/hentaifox all hit this).
  // Fix uniformly, regardless of the filter's type or inject mode: any shelf
  // that owns filters is surfaced as an always-visible picker, and choosing a
  // value both jumps into that shelf and applies the filter. Plain content
  // shelves stay tabs; source-level filters (e-hentai, mangadex, nhentai…) stay
  // inline exactly as before.
  function shelfOwnFilters(shelfId: string): Filter[] {
    return source?.manifest.endpoints?.shelves?.find((s) => s.id === shelfId)?.filters ?? [];
  }
  // A shelf with its own filters becomes a picker when it's a dedicated
  // "browse-by" view: either not the default shelf (genre/tag/catalog listed
  // after the content shelves) or driven by a required path filter. The default
  // shelf stays a content tab even if it carries an optional filter (e.g.
  // baozimh "popular" + genre), which then shows inline while that tab is active.
  const pickerShelves = $derived(
    shelves.flatMap((s, i) => {
      const filters = shelfOwnFilters(s.id);
      const isPicker = filters.length > 0 && (i > 0 || filters.some((f) => f.inject === 'path'));
      return isPicker ? [{ shelf: s, filters }] : [];
    }),
  );
  const tabShelves = $derived(shelves.filter((s) => !pickerShelves.some((p) => p.shelf.id === s.id)));
  const pickerFilterIds = $derived(new Set(pickerShelves.flatMap((p) => p.filters.map((f) => f.id))));
  // Inline filters = the active target's filters minus any surfaced as a picker
  // (so they aren't shown twice). The query still assembles from `activeFilters`.
  const inlineFilters = $derived(activeFilters.filter((f) => !pickerFilterIds.has(f.id)));

  const browsePageSize = $derived(
    source && target
      ? (typeof target === 'object' ? searchPageSize(source.manifest) : shelfPageSize(source.manifest, target))
      : Infinity,
  );
  const paginated = $derived(browsePageSize !== Infinity);

  // ── Infinite query (replaces manual loading/pagination/$effect) ──

  const browseQuery = createInfiniteQuery(() => ({
    queryKey: ['explore', source?.manifest.id, target, debouncedQuery, assembled.params, assembled.path, assembled.query] as const,
    queryFn: ({ pageParam }: { pageParam: number }) => {
      const src = source!;
      const t = target!;
      return fetchBrowse(src.manifest, t, {
        page: pageParam,
        q: debouncedQuery || undefined,
        filterParams: assembled.params,
        filterPath: assembled.path,
        filterQuery: assembled.query,
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
    // Keep the current grid on screen while switching source/shelf/filter so the
    // list dims-and-swaps instead of blanking out. (#3)
    placeholderData: keepPreviousData,
    // Browse listings change slowly and the gateway caches them for ~1h; a 5min
    // TanStack staleTime forced needless refetches on every back/tab-return. (#4)
    staleTime: 30 * 60_000,
  }));

  // Dedupe across pages: "latest"-style listings can re-bump a series onto a
  // later page, so the same item can appear twice. The keyed {#each} below uses
  // `manga.id`, and a duplicate key throws in Svelte 5 — which silently breaks
  // "load more" — so collapse on the same key the each block uses.
  const items = $derived(dedupeBy(browseQuery.data?.pages.flat() ?? [], (it) => it.id || it.url));

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
    }, { rootMargin: '1200px 0px' });
    observer.observe(node);
    return () => observer.disconnect();
  });

  // ── Source / shelf init ───────────────────────────────────────

  onMount(() => {
    const nextSources = listSources();
    allSources = nextSources;
    scrollEl = rootEl?.closest('main') ?? null;
    // A snapshot restore (back/forward) sets sourceId before this runs; only seed
    // defaults on a genuine fresh open or when the remembered source is gone.
    const restored = sourceId && nextSources.some((s) => s.enabled && s.manifest.id === sourceId);
    if (!restored) {
      const first = nextSources.find((s) => s.enabled) ?? null;
      sourceId = first?.manifest.id ?? '';
      if (first) {
        shelfId = getShelves(first.manifest)[0]?.id ?? '';
        filterState = getDefaultFilterState(first.manifest);
      }
    }
  });

  // Safety net for the old reset-on-source-change effect: if the remembered source
  // was disabled/removed while away, fall back to the first available one (which
  // also resets the view to that source's defaults).
  $effect(() => {
    if (sources.length && sourceId && !sources.some((s) => s.manifest.id === sourceId)) {
      selectSource(sources[0]!);
    }
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

  function selectSource(next: InstalledSource): void {
    if (next.manifest.id !== sourceId) {
      sourceId = next.manifest.id;
      // Switching to a different source resets the view to that source's defaults.
      shelfId = getShelves(next.manifest)[0]?.id ?? '';
      query = '';
      debouncedQuery = '';
      filterState = getDefaultFilterState(next.manifest);
    }
    trackSourceSelect({
      context: 'explore',
      source_id: next.manifest.id,
      source_name: next.manifest.name,
    });
  }

  // Preserve the browse view across navigation (open a manga → back): SvelteKit
  // calls capture before leaving and restore on back/forward. TanStack Query keeps
  // the loaded pages cached under the same query key, so restoring source/shelf/
  // query/filters re-shows the full list instantly; we then re-apply the scroll.
  export const snapshot = {
    capture: () => ({
      sourceId,
      shelfId,
      query,
      filterState: $state.snapshot(filterState),
      scrollTop: scrollEl?.scrollTop ?? 0,
    }),
    restore: (v: {
      sourceId: string;
      shelfId: string;
      query: string;
      filterState: Record<string, string | string[]>;
      scrollTop: number;
    }): void => {
      sourceId = v.sourceId;
      shelfId = v.shelfId;
      query = v.query;
      debouncedQuery = v.query.trim();
      filterState = v.filterState ?? {};
      pendingScrollTop = v.scrollTop;
    },
  };

  // Re-apply the restored scroll once the (cached) list has rendered its height.
  // Waiting on isPending + items keeps us from scrolling before the rows exist,
  // which would clamp to the short placeholder height.
  $effect(() => {
    if (pendingScrollTop == null || !scrollEl || browseQuery.isPending) return;
    void items.length;
    const el = scrollEl;
    const top = pendingScrollTop;
    pendingScrollTop = null;
    requestAnimationFrame(() => {
      el.scrollTop = top;
    });
  });

  async function openSourceSettings(): Promise<void> {
    await goto('/settings?tab=sources');
  }

  async function openManga(manga: MangaSummary): Promise<void> {
    if (!source || pendingMangaKey) return;
    const key = mangaKey(manga);
    pendingMangaKey = key;
    openError = '';
    trackSourceOpen({
      context: 'explore',
      source_id: source.manifest.id,
      source_name: source.manifest.name,
      shelf_id: typeof target === 'string' ? target : 'search',
      has_query: !!debouncedQuery,
    });
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

<div bind:this={rootEl} class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">
  {#if sources.length === 0}
    <div class="max-w-3xl mx-auto px-4 sm:px-6 py-10">
      <EmptyState title="Chưa có nguồn nào" hint="Mở Cài đặt → Nguồn để bật ít nhất một nguồn." />
      {#if disabledSourceCount > 0}
        <div class="-mt-10 flex justify-center">
          <button type="button" onclick={openSourceSettings}
            class="inline-flex items-center gap-1.5 h-8 px-3 rounded-full border border-dashed border-border-soft bg-surface text-xs font-medium text-text-muted transition-colors hover:bg-hover hover:text-text cursor-pointer"
          >
            <Plus size={12} /> Thêm nguồn
          </button>
        </div>
      {/if}
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
      <div class="flex flex-wrap gap-2" aria-label="Nguồn truyện">
        {#each sources as s (s.manifest.id)}
          {@const selected = s.manifest.id === sourceId}
          <button type="button" aria-pressed={selected}
            onclick={() => { selectSource(s); }}
            class={cn('inline-flex items-center gap-2 h-7 px-3 rounded-full chrome-label font-medium transition-colors',
              selected ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >{s.manifest.name}</button>
        {/each}
        {#if disabledSourceCount > 0}
          <button type="button" onclick={openSourceSettings}
            class="inline-flex items-center gap-1.5 h-7 px-3 rounded-full border border-dashed border-border-soft bg-surface chrome-label font-medium text-text-muted transition-colors hover:bg-hover hover:text-text cursor-pointer"
          >
            <Plus size={12} /> Thêm nguồn
          </button>
        {/if}
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

    {#if (!debouncedQuery && shelves.length > 1) || inlineFilters.length > 0}
      <!-- One controls row: view tabs, then any "browse-by" pickers (genre / tag
           / catalog — always visible, pick one and it jumps into that view),
           then a divider and the remaining inline filters. -->
      <div class="flex flex-wrap items-center gap-2">
        {#if !debouncedQuery && shelves.length > 1}
          {#each tabShelves as shelf (shelf.id)}
            {@const selected = shelf.id === shelfId}
            <button type="button" onclick={() => { shelfId = shelf.id; }}
              class={cn('inline-flex items-center h-7 px-3 rounded-full chrome-label font-medium transition-colors',
                selected ? 'bg-surface-2 text-text' : 'text-text-muted hover:text-text',
              )}
            >{shelf.label}</button>
          {/each}
          {#each pickerShelves as p (p.shelf.id)}
            {#each p.filters as filter (filter.id)}
              <FilterGroup
                {filter}
                selection={filterState}
                onChange={(next) => { filterState = next; shelfId = p.shelf.id; }}
              />
            {/each}
          {/each}
        {/if}
        {#if inlineFilters.length > 0}
          {#if !debouncedQuery && shelves.length > 1}
            <span class="h-5 w-px shrink-0 bg-divider" aria-hidden="true"></span>
          {/if}
          <FilterChips filters={inlineFilters} selection={filterState} onChange={(next) => { filterState = next; }} />
        {/if}
      </div>
    {/if}

    {#if browseQuery.isPending}
      <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-x-3 gap-y-4 sm:gap-x-4 sm:gap-y-5">
        <CardSkeleton count={18} />
      </div>
    {:else if browseQuery.error && items.length === 0}
      <div class="py-12 flex flex-col items-center gap-3 text-center">
        <div>
          <p class="text-sm font-medium text-error-text">Không tải được</p>
          <p class="text-xs text-text-subtle mt-1">{browseQuery.error.message}</p>
        </div>
        <button type="button" onclick={() => browseQuery.refetch()}
          class="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-surface-2 text-text text-sm font-medium hover:bg-interactive-hover transition-colors cursor-pointer"
        >
          <RotateCw size={14} /> Thử lại
        </button>
      </div>
    {:else if items.length === 0}
      <div class="py-12 text-center">
        <p class="text-sm font-medium text-text">{debouncedQuery ? 'Không tìm thấy' : 'Chưa có truyện ở đây'}</p>
        {#if debouncedQuery}<p class="text-xs text-text-subtle mt-1">Thử từ khoá khác.</p>{/if}
      </div>
    {:else}
      <div class={cn('grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-x-3 gap-y-4 sm:gap-x-4 sm:gap-y-5 transition-opacity', browseQuery.isPlaceholderData && 'opacity-50 pointer-events-none')}>
        {#each items as manga (manga.id)}
          {@const existing = existingByUrl[manga.url]}
          {@const key = mangaKey(manga)}
          {@const pending = pendingMangaKey === key}
          <WorkCard
            class="cv-card"
            work={{
              title: manga.title,
              cover_url: manga.cover,
              coverHeaders: manga.coverHeaders,
              chapter: manga.latestChapter,
              nsfw: source?.manifest.nsfw,
              inLibrary: existing?.in_library,
            }}
            onclick={() => openManga(manga)}
            {pending}
            disabled={pendingMangaKey !== null}
            dimmed={pendingMangaKey !== null && !pending}
          />
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
