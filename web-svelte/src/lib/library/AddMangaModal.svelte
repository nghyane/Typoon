<script lang="ts">
  import { goto } from '$app/navigation';
  import { AlertTriangle, Check, CheckCircle2, ChevronDown, Link as LinkIcon, Search, Wand2 } from 'lucide-svelte';
  import { addWorkToLibrary, createBlankWork, ensureWorkFromSource } from '$lib/works/repo';
  import { fetchBrowse, fetchMangaDetail } from '$lib/source/runtime/endpoints';
  import { hasSearch } from '$lib/source/runtime/metadata';
  import { listSources, setSourceEnabled } from '$lib/source/registry';
  import type { InstalledSource, MangaDetail, MangaSummary } from '$lib/source/types';
  import { cn } from '$lib/cn';
  import Button from '$lib/ui/Button.svelte';
  import Cover from '$lib/ui/Cover.svelte';
  import Modal from '$lib/ui/Modal.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';

  let { open, onClose }: { open: boolean; onClose: () => void } = $props();

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
  const BLANK_TITLE_FALLBACK = 'Manga mới';

  let query = $state('');
  let debouncedQuery = $state('');
  let sources = $state<InstalledSource[]>([]);
  let hits = $state<SearchHit[]>([]);
  let failures = $state<SearchFailure[]>([]);
  let loading = $state(false);
  let pendingKey = $state<string | null>(null);
  let error = $state('');
  let scopeId = $state<string | null>(null);
  let expandedBySource = $state<Record<string, boolean>>({});
  let urlImportKey = $state('');
  let urlError = $state('');
  let urlBusy = $state(false);

  const enabledSources = $derived(sources.filter((source) => source.enabled));
  const searchableSources = $derived(enabledSources.filter((source) => hasSearch(source.manifest)));
  const trimmed = $derived(query.trim());
  const isUrl = $derived(isUrlLike(trimmed));
  const urlMatch = $derived(isUrl ? matchSource(trimmed, enabledSources) : null);
  const busy = $derived(pendingKey !== null);
  const scopedHits = $derived(scopeId === null ? hits : hits.filter((hit) => hit.source.manifest.id === scopeId));
  const visibleSources = $derived(scopeId === null
    ? searchableSources
    : searchableSources.filter((source) => source.manifest.id === scopeId));
  const hitCounts = $derived.by(() => {
    const counts = new Map<string, number>();
    for (const hit of hits) {
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

  $effect(() => {
    if (!open) return;
    sources = listSources();
    query = '';
    debouncedQuery = '';
    hits = [];
    failures = [];
    loading = false;
    pendingKey = null;
    error = '';
    scopeId = null;
    expandedBySource = {};
    urlImportKey = '';
    urlError = '';
    urlBusy = false;
  });

  $effect(() => {
    if (!open) return;
    const value = query;
    const timer = window.setTimeout(() => { debouncedQuery = value.trim(); }, 250);
    return () => window.clearTimeout(timer);
  });

  $effect(() => {
    const q = debouncedQuery.trim();
    if (!open || isUrlLike(q) || q.length < 2) {
      hits = [];
      failures = [];
      loading = false;
      return;
    }

    let cancelled = false;
    loading = true;
    error = '';

    Promise.allSettled(
      searchableSources.map(async (source) => ({
        source,
        items: await fetchBrowse(source.manifest, { search: true as const }, { page: 1, q }),
      })),
    ).then((results) => {
      if (cancelled) return;
      const nextHits: SearchHit[] = [];
      const nextFailures: SearchFailure[] = [];
      for (const result of results) {
        if (result.status === 'fulfilled') {
          nextHits.push(...rankAndCap(q, result.value.source, result.value.items));
        } else {
          nextFailures.push({ sourceId: 'unknown', error: errorFrom(result.reason) });
        }
      }
      hits = nextHits;
      failures = nextFailures;
    }).catch((err: Error) => {
      if (!cancelled) error = err.message;
    }).finally(() => {
      if (!cancelled) loading = false;
    });

    return () => { cancelled = true; };
  });

  $effect(() => {
    const match = urlMatch;
    if (!open || !isUrl || !match) {
      urlImportKey = '';
      urlError = '';
      urlBusy = false;
      return;
    }

    const key = hitKey({ source: match.source, manga: { id: match.upstreamRef, url: match.upstreamRef, title: match.upstreamRef, cover: null }, score: 1 });
    if (urlImportKey === key) return;

    let cancelled = false;
    urlImportKey = key;
    urlError = '';
    urlBusy = true;

    fetchMangaDetail(match.source.manifest, match.upstreamRef)
      .then((detail) => {
        if (cancelled) return;
        void importHit({
          source: match.source,
          manga: { id: match.upstreamRef, url: match.upstreamRef, title: detail.title, cover: detail.cover },
          score: 1,
        }, detail, key);
      })
      .catch((err) => {
        if (!cancelled) urlError = errorFrom(err).message;
      })
      .finally(() => {
        if (!cancelled) urlBusy = false;
      });

    return () => { cancelled = true; };
  });

  function setQueryValue(value: string): void {
    query = value;
    scopeId = null;
    expandedBySource = {};
    error = '';
    urlError = '';
    urlImportKey = '';
  }

  function refreshSources(): void {
    sources = listSources();
  }

  function toggleSource(source: InstalledSource): void {
    if (!hasSearch(source.manifest)) return;
    setSourceEnabled(source.manifest.id, !source.enabled);
    refreshSources();
  }

  async function importHit(hit: SearchHit, detail?: MangaDetail | null, keyOverride?: string): Promise<void> {
    const key = keyOverride ?? hitKey(hit);
    if (pendingKey && pendingKey !== key) return;
    pendingKey = key;
    error = '';
    try {
      const resolved = detail ?? await fetchMangaDetail(hit.source.manifest, hit.manga.url).catch(() => null);
      const work = await ensureWorkFromSource(hit.source.manifest, resolved ?? hit.manga);
      if (!work.in_library) await addWorkToLibrary(work.id);
      onClose();
    } catch (err) {
      error = errorFrom(err).message;
    } finally {
      if (pendingKey === key) pendingKey = null;
    }
  }

  async function importBlank(title: string): Promise<void> {
    if (pendingKey) return;
    pendingKey = 'blank';
    error = '';
    try {
      const work = await createBlankWork({
        title: title.trim() || BLANK_TITLE_FALLBACK,
        target_lang: 'vi',
      });
      await addWorkToLibrary(work.id);
      onClose();
      await goto(`/w/${work.id}`);
    } catch (err) {
      error = errorFrom(err).message;
    } finally {
      if (pendingKey === 'blank') pendingKey = null;
    }
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

  function isUrlLike(input: string): boolean {
    return /^https?:\/\//i.test(input.trim());
  }

  function matchSource(raw: string, list: InstalledSource[]): { source: InstalledSource; upstreamRef: string } | null {
    let parsed: URL;
    try { parsed = new URL(raw.trim()); } catch { return null; }
    const host = parsed.host.toLowerCase();
    const source = list.find((item) => item.manifest.host.toLowerCase() === host)
      ?? list.find((item) => host.endsWith(`.${item.manifest.host.toLowerCase()}`));
    return source ? { source, upstreamRef: raw.trim() } : null;
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

<Modal open={open} onClose={onClose} title="Thêm manga vào thư viện" size="md">
  <div class="px-5 py-4 space-y-3 min-h-[420px]">
    <div class="relative">
      {#if isUrl}
        <LinkIcon size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
      {:else}
        <Search size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
      {/if}
      <input
        type="text"
        value={query}
        oninput={(event) => setQueryValue(event.currentTarget.value)}
        disabled={busy}
        placeholder="Tìm tên truyện hoặc dán đường dẫn manga"
        class={cn(inputCls, 'pl-9 h-10', isUrl && (urlMatch ? 'pr-36' : 'pr-32'))}
      />
      {#if isUrl}
        <span class={cn('absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 h-6 px-2 rounded-xs text-xs font-medium pointer-events-none', urlMatch ? 'bg-success-bg text-success-text' : 'bg-warning-bg text-warning-text')}>
          {#if urlMatch}<CheckCircle2 size={12} />{urlMatch.source.manifest.name}{:else}<AlertTriangle size={12} />Chưa hỗ trợ{/if}
        </span>
      {/if}
    </div>

    {#if isUrl}
      {#if urlMatch}
        {#if urlError}
          <div class="rounded-md bg-error-bg border border-error-text/20 px-4 py-3">
            <div class="flex items-start gap-3">
              <AlertTriangle size={14} class="text-error-text shrink-0 mt-0.5" />
              <div class="flex-1 min-w-0">
                <p class="text-sm text-text">Không tải được từ {urlMatch.source.manifest.name}</p>
                <p class="text-xs text-error-text mt-1 line-clamp-2">{urlError}</p>
              </div>
            </div>
          </div>
        {:else}
          <div class="rounded-md bg-surface-2 px-4 py-3 flex items-center gap-3">
            <Spinner size={14} class="text-info-text shrink-0" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-text">{pendingKey ? 'Đang thêm vào thư viện…' : `Đang tải từ ${urlMatch.source.manifest.name}…`}</p>
              <p class="text-xs text-text-subtle truncate mt-1">{urlMatch.upstreamRef}</p>
            </div>
          </div>
        {/if}
      {:else}
        <div class="rounded-md bg-warning-bg border border-warning-text/20 px-4 py-3">
          <div class="flex items-start gap-3">
            <AlertTriangle size={14} class="text-warning-text shrink-0 mt-0.5" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-text">Không có nguồn quản lý site này</p>
              <p class="text-xs text-text-subtle mt-1 break-all line-clamp-2">{trimmed}</p>
              <Button variant="secondary" size="sm" onclick={() => importBlank('')} disabled={busy} class="mt-3">
                {#if pendingKey === 'blank'}<Spinner size={14} />{:else}<Wand2 size={14} />{/if}
                Tạo trống thay
              </Button>
            </div>
          </div>
        </div>
      {/if}
    {:else if debouncedQuery.length < 2}
      {#if sources.length === 0}
        <div class="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
          <p class="text-sm text-text-muted">Chưa cài nguồn nào</p>
          <p class="text-xs text-text-subtle mt-1">Mở Cài đặt để cài nguồn đầu tiên.</p>
        </div>
      {:else}
        <div class="space-y-2">
          <p class="text-xs text-text-subtle px-0.5">Bấm để bật/tắt nguồn cho fanout search</p>
          <ul class="flex flex-wrap gap-2">
            {#each sources as source (source.manifest.id)}
              {@const searchable = hasSearch(source.manifest)}
              {@const enabled = source.enabled && searchable}
              <li>
                <button
                  type="button"
                  onclick={() => toggleSource(source)}
                  disabled={!searchable || busy}
                  title={searchable ? (enabled ? `Tắt ${source.manifest.name}` : `Bật ${source.manifest.name}`) : `${source.manifest.name} chưa hỗ trợ tìm — dán link để thêm`}
                  class={cn(
                    'inline-flex items-center gap-2 h-8 pl-2 pr-3 rounded-sm text-xs transition-colors',
                    !searchable
                      ? 'bg-surface-2 text-text-subtle cursor-not-allowed border border-border-soft opacity-50'
                      : enabled
                      ? 'bg-accent-bg text-text border border-accent-text/30 hover:brightness-110 cursor-pointer'
                      : 'bg-surface-2 text-text-muted border border-border-soft hover:bg-hover hover:text-text cursor-pointer',
                  )}
                >
                  <span class={cn('size-1.5 rounded-full shrink-0', enabled ? 'bg-accent' : searchable ? 'bg-text-subtle/40' : 'bg-text-subtle/20')}></span>
                  <span class="font-medium truncate max-w-[140px]">{source.manifest.name}</span>
                  <span class="text-xs text-text-subtle truncate">{source.manifest.host}</span>
                </button>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    {:else}
      {#if hits.length > 0 && (sourcesWithHits.length > 1 || scopeId !== null)}
        <div class="flex items-center gap-1 overflow-x-auto px-0.5" style="scrollbar-width: none">
          <button type="button" onclick={() => { scopeId = null; }} class={cn('inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0 transition-colors', scopeId === null ? 'bg-surface-2 text-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
            Tất cả <span class={cn('text-xs tabular-nums', scopeId === null ? 'text-text-subtle' : 'text-text-subtle/70')}>{hits.length}</span>
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
                  <li>
                    <button
                      type="button"
                      onclick={() => importHit(hit)}
                      disabled={busy}
                      class={cn(
                        'w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-hover transition-colors cursor-pointer',
                        pendingKey === key && 'opacity-60 cursor-wait',
                        busy && pendingKey !== key && 'opacity-60 cursor-not-allowed',
                      )}
                    >
                      <Cover src={hit.manga.cover} headers={hit.manga.coverHeaders} title={hit.manga.title} class="w-8 aspect-[2/3] rounded-xs shrink-0" fontSize="text-xs" />
                      <div class="flex-1 min-w-0">
                        <p class="text-sm text-text truncate leading-tight">{hit.manga.title}</p>
                        {#if hit.source.manifest.languages.length > 0}
                          <p class="text-xs text-text-subtle uppercase mt-1">{hit.source.manifest.languages.slice(0, 3).join('/')}</p>
                        {/if}
                      </div>
                      {#if pendingKey === key}<Spinner size={14} class="text-text-subtle shrink-0" />{/if}
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

      <button
        type="button"
        onclick={() => importBlank(query)}
        disabled={busy}
        class={cn(
          'w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-colors cursor-pointer disabled:cursor-wait disabled:opacity-60',
          scopedHits.length === 0 ? 'bg-accent-bg hover:brightness-110' : 'bg-surface-2 hover:bg-hover',
        )}
      >
        <span class={cn('inline-flex items-center justify-center size-8 rounded-sm shrink-0', scopedHits.length === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted')}>
          {#if pendingKey === 'blank'}<Spinner size={14} />{:else}<Wand2 size={14} />{/if}
        </span>
        <div class="flex-1 min-w-0">
          <p class="text-sm text-text">
            {#if scopedHits.length === 0}
              {trimmed ? `Không tìm thấy. Tạo "${trimmed}" trống?` : 'Tạo manga trống'}
            {:else}
              {trimmed ? `Không thấy "${trimmed}"? Tạo trống` : 'Tạo manga trống'}
            {/if}
          </p>
          <p class="text-xs text-text-subtle mt-1">Vào trang truyện để liên kết nguồn đọc sau.</p>
        </div>
      </button>
    {/if}

    {#if error}
      <p class="text-sm text-error-text text-center">{error}</p>
    {/if}
  </div>

  {#snippet footerLeft()}
    {sources.length} nguồn đã cài
  {/snippet}

  {#snippet footer()}
    <Button variant="ghost" onclick={onClose} disabled={busy}>Huỷ</Button>
  {/snippet}
</Modal>
