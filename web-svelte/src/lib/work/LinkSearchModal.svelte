<script lang="ts">
  import { AlertTriangle, Check, ChevronDown, Search } from 'lucide-svelte';
  import { attachSource } from '$lib/works/repo';
  import { fetchBrowse, fetchMangaDetail } from '$lib/source/runtime/endpoints';
  import { hasSearch } from '$lib/source/runtime/metadata';
  import { listSources } from '$lib/source/registry';
  import type { WorkSource } from '$lib/db';
  import type { InstalledSource, MangaDetail, MangaSummary } from '$lib/source/types';
  import { cn } from '$lib/cn';
  import Button from '$lib/ui/Button.svelte';
  import Cover from '$lib/ui/Cover.svelte';
  import Modal from '$lib/ui/Modal.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';

  let {
    open,
    onClose,
    workId,
    workTitle,
    ownSources,
    onLinked,
  }: {
    open: boolean;
    onClose: () => void;
    workId: string;
    workTitle: string;
    ownSources: WorkSource[];
    onLinked?: () => void | Promise<void>;
  } = $props();

  interface SearchHit {
    source: InstalledSource;
    manga: MangaSummary;
    score: number;
  }

  interface SearchFailure {
    sourceId: string;
    error: Error;
  }

  const INITIAL_PREVIEW = 3;
  const PER_GROUP_MAX = 8;

  let query = $state('');
  let debouncedQuery = $state('');
  let sources = $state(listSources());

  // Sync query from prop on open
  $effect(() => {
    query = workTitle;
    debouncedQuery = workTitle.trim();
  });
  let hits = $state<SearchHit[]>([]);
  let failures = $state<SearchFailure[]>([]);
  let loading = $state(false);
  let pendingKey = $state<string | null>(null);
  let pickedKeys = $state<Set<string>>(new Set());
  let scopeId = $state<string | null>(null);
  let expandedBySource = $state<Record<string, boolean>>({});
  let error = $state('');
  let inputEl = $state<HTMLInputElement | null>(null);
  let searchSeq = 0;

  const searchableSources = $derived(sources.filter((source) => source.enabled && hasSearch(source.manifest)));
  const ownKeys = $derived.by(() => {
    const keys = new Set<string>();
    for (const source of ownSources) keys.add(`${source.source}::${source.upstream_ref}`);
    return keys;
  });
  const visibleHits = $derived(hits.filter((hit) => !ownKeys.has(`${hit.source.manifest.id}::${hit.manga.url}`)));
  const scopedHits = $derived(scopeId === null ? visibleHits : visibleHits.filter((hit) => hit.source.manifest.id === scopeId));
  const visibleSources = $derived(scopeId === null
    ? searchableSources
    : searchableSources.filter((source) => source.manifest.id === scopeId));
  const hitCounts = $derived.by(() => {
    const counts = new Map<string, number>();
    for (const hit of visibleHits) {
      const id = hit.source.manifest.id;
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    return counts;
  });
  const sourcesWithHits = $derived(searchableSources.filter((source) => hitCounts.has(source.manifest.id)));
  const resultGroups = $derived.by(() => {
    const by = new Map<string, { source: InstalledSource; hits: SearchHit[] }>();
    for (const hit of scopedHits) {
      const id = hit.source.manifest.id;
      if (!by.has(id)) by.set(id, { source: hit.source, hits: [] });
      by.get(id)!.hits.push(hit);
    }
    for (const group of by.values()) group.hits.sort((a, b) => b.score - a.score);
    return visibleSources
      .map((source) => by.get(source.manifest.id))
      .filter((group): group is { source: InstalledSource; hits: SearchHit[] } => !!group);
  });
  const singleSourceResults = $derived(visibleSources.length === 1);

  // Focus input on mount (component is conditionally mounted)
  $effect(() => {
    inputEl?.focus();
  });

  // Debounce query
  $effect(() => {
    const value = query;
    const timer = setTimeout(() => { debouncedQuery = value.trim(); }, 250);
    return () => clearTimeout(timer);
  });

  // Fanout search — triggers on mount (initial debouncedQuery) and on every keystroke
  $effect(() => {
    const q = debouncedQuery.trim();
    if (q.length < 2) {
      hits = [];
      failures = [];
      loading = false;
      return;
    }

    let cancelled = false;
    loading = true;
    error = '';

    const targets = sources.filter((source) => source.enabled && hasSearch(source.manifest));
    const seq = ++searchSeq;

    Promise.all(targets.map(async (source) => {
      try {
        return {
          source,
          items: await fetchBrowse(source.manifest, { search: true as const }, { page: 1, q }),
          error: null,
        };
      } catch (err) {
        return { source, items: [] as MangaSummary[], error: errorFrom(err) };
      }
    })).then((results) => {
      if (cancelled || seq !== searchSeq) return;
      const nextHits: SearchHit[] = [];
      const nextFailures: SearchFailure[] = [];
      for (const result of results) {
        if (result.error) nextFailures.push({ sourceId: result.source.manifest.id, error: result.error });
        else nextHits.push(...rankAndCap(q, result.source, result.items));
      }
      hits = nextHits;
      failures = nextFailures;
    }).catch((err: Error) => {
      if (!cancelled && seq === searchSeq) error = err.message;
    }).finally(() => {
      if (!cancelled && seq === searchSeq) loading = false;
    });

    return () => { cancelled = true; };
  });

  function close(): void {
    pendingKey = null;
    pickedKeys = new Set();
    onClose();
  }

  function setQueryValue(value: string): void {
    query = value;
    scopeId = null;
    expandedBySource = {};
    error = '';
  }

  async function pickHit(hit: SearchHit): Promise<void> {
    const key = hitKey(hit);
    if (pendingKey || pickedKeys.has(key)) return;
    pendingKey = key;
    error = '';
    try {
      const detail = await fetchMangaDetail(hit.source.manifest, hit.manga.url).catch(() => null);
      const source = toWorkSource(hit, detail);
      await attachSource(workId, source);
      pickedKeys = new Set(pickedKeys).add(key);
      await onLinked?.();
    } catch (err) {
      error = `Liên kết thất bại: ${errorFrom(err).message}`;
    } finally {
      if (pendingKey === key) pendingKey = null;
    }
  }

  function toWorkSource(hit: SearchHit, detail: MangaDetail | null): WorkSource {
    const manifest = hit.source.manifest;
    return {
      source: manifest.id,
      upstream_ref: hit.manga.url,
      title: detail?.title ?? hit.manga.title,
      cover_url: detail?.cover ?? hit.manga.cover ?? null,
      languages: detail?.availableLanguages ?? manifest.languages ?? [],
      added_at: new Date().toISOString(),
    };
  }

  function expandSource(id: string): void {
    expandedBySource = { ...expandedBySource, [id]: true };
  }

  function hitKey(hit: SearchHit): string {
    return `${hit.source.manifest.id}::${hit.manga.id}`;
  }

  function errorFrom(value: unknown): Error {
    return value instanceof Error ? value : new Error(String(value));
  }

  function fuzzyScore(search: string, title: string): number {
    const q = search.trim().toLowerCase();
    const t = title.trim().toLowerCase();
    if (!q || !t) return 0;
    if (t === q) return 1;
    if (t.startsWith(q)) return 0.95;
    if (t.includes(q)) return 0.85;

    const qTokens = q.split(/\s+/).filter(Boolean);
    const tTokens = t.split(/\s+/).filter(Boolean);
    if (qTokens.length === 0) return 0;

    let matched = 0;
    for (const token of qTokens) {
      if (tTokens.some((item) => item.includes(token))) matched += 1;
    }
    const overlap = matched / qTokens.length;

    const qBigrams = new Set(bigrams(q));
    const tBigrams = bigrams(t);
    if (qBigrams.size === 0 || tBigrams.length === 0) return overlap * 0.7;
    let bigramHit = 0;
    for (const bigram of tBigrams) if (qBigrams.has(bigram)) bigramHit += 1;
    const bigramOverlap = bigramHit / Math.max(qBigrams.size, tBigrams.length);
    return Math.max(overlap * 0.7, bigramOverlap * 0.6);
  }

  function bigrams(value: string): string[] {
    const out: string[] = [];
    for (let i = 0; i < value.length - 1; i += 1) out.push(value.slice(i, i + 2));
    return out;
  }

  function rankAndCap(search: string, source: InstalledSource, items: MangaSummary[]): SearchHit[] {
    const scored = items.map((manga) => ({ source, manga, score: fuzzyScore(search, manga.title) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, PER_GROUP_MAX);
  }

  const inputCls = 'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors';
</script>

<Modal open={open} onClose={close} title="Liên kết nguồn" size="md">
  <div class="px-5 py-4 space-y-3 min-h-[420px]">
    <div class="relative">
      <Search size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
      <input
        bind:this={inputEl}
        type="text"
        value={query}
        oninput={(event) => setQueryValue(event.currentTarget.value)}
        disabled={pendingKey !== null}
        placeholder="Tìm tên truyện ở nguồn khác"
        class={cn(inputCls, 'pl-9 h-10')}
      />
    </div>

    {#if query.trim().length < 2}
      <p class="text-sm text-text-subtle px-0.5">
        Gõ ít nhất 2 ký tự để bắt đầu tìm trên {searchableSources.length} nguồn.
      </p>
    {:else}
      {#if visibleHits.length > 0 && (sourcesWithHits.length > 1 || scopeId !== null)}
        <div class="flex items-center gap-1 overflow-x-auto px-0.5" style="scrollbar-width: none">
          <button type="button" onclick={() => { scopeId = null; }} class={cn('inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0 transition-colors', scopeId === null ? 'bg-surface-2 text-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
            Tất cả <span class={cn('text-xs tabular-nums', scopeId === null ? 'text-text-subtle' : 'text-text-subtle/70')}>{visibleHits.length}</span>
          </button>
          {#each sourcesWithHits as source (source.manifest.id)}
            <button type="button" onclick={() => { scopeId = source.manifest.id; }} class={cn('inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0 transition-colors', scopeId === source.manifest.id ? 'bg-surface-2 text-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
              {source.manifest.name}
              <span class={cn('text-xs tabular-nums', scopeId === source.manifest.id ? 'text-text-subtle' : 'text-text-subtle/70')}>{hitCounts.get(source.manifest.id) ?? 0}</span>
            </button>
          {/each}
        </div>
      {/if}

      {#if loading && scopedHits.length === 0}
        <div class="flex items-center gap-3 px-4 py-3 rounded-md bg-surface-2">
          <Spinner size={14} class="text-info-text" />
          <p class="text-sm text-text-muted">Đang tìm trên {visibleSources.length} nguồn…</p>
        </div>
      {:else}
        <div class="space-y-3">
          <div class="flex items-center justify-between gap-2 px-0.5">
            <p class="text-xs uppercase tracking-wider text-text-subtle">
              {scopedHits.length} kết quả{#if loading}<span class="ml-1.5 normal-case">· đang tìm thêm…</span>{/if}
            </p>
            {#if failures.length > 0}
              <span class="text-xs text-warning-text inline-flex items-center gap-1"><AlertTriangle size={12} />{failures.length} nguồn lỗi</span>
            {/if}
          </div>

          {#each resultGroups as group (group.source.manifest.id)}
            {@const sourceId = group.source.manifest.id}
            {@const capped = group.hits.slice(0, PER_GROUP_MAX)}
            {@const visible = singleSourceResults || expandedBySource[sourceId] ? capped : capped.slice(0, INITIAL_PREVIEW)}
            {@const more = capped.length - visible.length}
            <section>
              {#if !singleSourceResults}
                <header class="flex items-baseline justify-between gap-2 px-1 mb-1.5">
                  <div class="flex items-baseline gap-2 min-w-0">
                    <span class="text-xs font-medium text-text truncate">{group.source.manifest.name}</span>
                    <span class="text-xs text-text-subtle truncate">{group.source.manifest.host}</span>
                  </div>
                  <span class="text-xs text-text-subtle shrink-0">{group.hits.length}</span>
                </header>
              {/if}
              <ul class="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
                {#each visible as hit (hitKey(hit))}
                  {@const key = hitKey(hit)}
                  {@const picked = pickedKeys.has(key)}
                  {@const busy = pendingKey !== null}
                  <li>
                    <button
                      type="button"
                      onclick={() => pickHit(hit)}
                      disabled={busy || picked}
                      class={cn(
                        'w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-hover transition-colors cursor-pointer',
                        picked && 'opacity-70 cursor-default',
                        pendingKey === key && 'opacity-60 cursor-wait',
                        busy && pendingKey !== key && !picked && 'opacity-60 cursor-not-allowed',
                      )}
                    >
                      <Cover src={hit.manga.cover} title={hit.manga.title} class="w-8 aspect-[2/3] rounded-xs shrink-0" fontSize="text-xs" />
                      <div class="flex-1 min-w-0">
                        <p class="text-sm text-text truncate leading-tight">{hit.manga.title}</p>
                        {#if hit.source.manifest.languages.length > 0}
                          <p class="text-xs text-text-subtle uppercase mt-1">{hit.source.manifest.languages.slice(0, 3).join('/')}</p>
                        {/if}
                      </div>
                      {#if pendingKey === key}<Spinner size={14} class="text-text-subtle shrink-0" />{/if}
                      {#if picked}<Check size={14} class="text-success-text shrink-0" />{/if}
                    </button>
                  </li>
                {/each}
                {#if more > 0}
                  <li>
                    <button type="button" onclick={() => expandSource(sourceId)} class="w-full inline-flex items-center justify-center gap-2 h-8 text-xs text-text-muted hover:bg-hover hover:text-text transition-colors cursor-pointer">
                      <ChevronDown size={12} /> Xem thêm {more}
                    </button>
                  </li>
                {/if}
              </ul>
            </section>
          {/each}
        </div>
      {/if}
    {/if}

    {#if error}
      <p class="text-sm text-error-text text-center">{error}</p>
    {/if}
  </div>

  {#snippet footerLeft()}
    Liên kết vào: {workTitle}{#if pickedKeys.size > 0} · đã thêm {pickedKeys.size}{/if}
  {/snippet}

  {#snippet footer()}
    <Button variant="ghost" onclick={close} disabled={pendingKey !== null}>Đóng</Button>
  {/snippet}
</Modal>
