<script lang="ts">
  import { browser } from '$app/environment';
  import { Plus, RotateCw, Search } from 'lucide-svelte';
  import { listLibraryWorks } from '$lib/works/repo';
  import { getWorkUpdatesMap, checkLibraryUpdates } from '$lib/works/updates';
  import type { LibraryStatus, Work, WorkUpdate } from '$lib/db';
  import { cn } from '$lib/cn';
  import AddMangaModal from '$lib/library/AddMangaModal.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import CardSkeleton from '$lib/ui/CardSkeleton.svelte';
  import WorkCard from '$lib/ui/WorkCard.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  type StatusFilter = 'all' | LibraryStatus;

  const statusTabs: Array<{ value: StatusFilter; label: string }> = [
    { value: 'all', label: 'Tất cả' },
    { value: 'reading', label: 'Đang đọc' },
    { value: 'plan', label: 'Định đọc' },
    { value: 'done', label: 'Đã đọc' },
    { value: 'dropped', label: 'Bỏ' },
  ];

  // Remember the reader's last status tab across navigations, reloads, and sessions
  // so coming back to the library lands on the shelf they were on, not always 'all'.
  const TAB_STORAGE_KEY = 'typoon.library.tab.v1';
  const isStatusFilter = (v: unknown): v is StatusFilter => statusTabs.some((tab) => tab.value === v);
  function loadSavedTab(): StatusFilter {
    if (!browser) return 'all';
    const saved = localStorage.getItem(TAB_STORAGE_KEY);
    return isStatusFilter(saved) ? saved : 'all';
  }

  type SortKey = 'recent' | 'title' | 'updated';
  const sortOptions: Array<{ value: SortKey; label: string }> = [
    { value: 'updated', label: 'Chương mới' },
    { value: 'recent', label: 'Mở gần đây' },
    { value: 'title', label: 'Tên A→Z' },
  ];

  let works = $state<Work[]>([]);
  let updates = $state<Record<string, WorkUpdate>>({});
  let status = $state<StatusFilter>(loadSavedTab());
  let query = $state('');
  let sort = $state<SortKey>('updated');
  let loading = $state(true);
  let checking = $state(false);
  let error = $state('');
  let addOpen = $state(false);

  async function load(): Promise<void> {
    loading = true;
    error = '';
    try {
      works = await listLibraryWorks();
      updates = await getWorkUpdatesMap(works.map((w) => w.id));
      void backgroundCheck();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  // Stale-only check on visit (TTL-gated, throttled in updates.ts) — keeps data
  // fresh without blocking render or hammering the gateway every open.
  async function backgroundCheck(): Promise<void> {
    try {
      const { withNew } = await checkLibraryUpdates();
      if (withNew > 0) updates = await getWorkUpdatesMap(works.map((w) => w.id));
    } catch {
      /* silent: a flaky source shouldn't surface here */
    }
  }

  // Manual full refresh — forces every followed title, with feedback.
  async function refreshUpdates(): Promise<void> {
    if (checking) return;
    checking = true;
    try {
      const { withNew } = await checkLibraryUpdates({ force: true });
      updates = await getWorkUpdatesMap(works.map((w) => w.id));
      toast.show({
        title: withNew > 0 ? `${withNew} bộ có chương mới` : 'Đã là mới nhất',
        variant: 'success',
        duration: 2500,
      });
    } catch (err) {
      toast.show({ title: 'Không kiểm tra được cập nhật', description: err instanceof Error ? err.message : undefined, variant: 'error' });
    } finally {
      checking = false;
    }
  }

  // A new chapter the user hasn't seen: the newest chapter appeared after they
  // last opened the work.
  function hasUpdate(work: Work): boolean {
    const u = updates[work.id];
    return !!u && !!work.last_opened_at && u.updated_at > work.last_opened_at;
  }

  const counts = $derived({
    all: works.length,
    reading: works.filter((work) => work.library_status === 'reading').length,
    plan: works.filter((work) => work.library_status === 'plan').length,
    done: works.filter((work) => work.library_status === 'done').length,
    dropped: works.filter((work) => work.library_status === 'dropped').length,
  });
  function sortWorks(list: Work[], key: SortKey): Work[] {
    const by = [...list];
    if (key === 'title') return by.sort((a, b) => a.title.localeCompare(b.title));
    if (key === 'updated') {
      // Newest chapter first: the detected chapter publish time, falling back to
      // the work's own updated_at until the feed has been checked.
      const stamp = (w: Work) => updates[w.id]?.updated_at ?? w.updated_at ?? '';
      return by.sort((a, b) => stamp(b).localeCompare(stamp(a)));
    }
    return by.sort((a, b) => String(b.last_opened_at ?? '').localeCompare(String(a.last_opened_at ?? '')));
  }

  const filtered = $derived(
    sortWorks(
      works.filter((work) => {
        if (status !== 'all' && work.library_status !== status) return false;
        const term = query.trim().toLowerCase();
        return !term || work.title.toLowerCase().includes(term);
      }),
      sort,
    ),
  );

  $effect(() => { void load(); });

  // Persist the active tab whenever it changes.
  $effect(() => {
    if (browser) localStorage.setItem(TAB_STORAGE_KEY, status);
  });
</script>

<svelte:head><title>Thư viện — Hội Mê Truyện</title></svelte:head>

{#if loading}
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-6">
    <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-x-3 gap-y-4 sm:gap-x-4 sm:gap-y-5">
      <CardSkeleton count={18} />
    </div>
  </div>
{:else if error}
  <ErrorState title="Không tải được thư viện" message={error} onRetry={load} retrying={loading} />
{:else if works.length === 0}
  <div class="max-w-3xl mx-auto px-4 sm:px-6 py-10">
    <EmptyState title="Thư viện trống" hint="Tìm truyện hoặc dán URL để thêm vào.">
      {#snippet action()}
        <Button variant="primary" size="md" onclick={() => { addOpen = true; }}>
          <Plus size={14} /> Thêm truyện
        </Button>
      {/snippet}
    </EmptyState>
  </div>
  <AddMangaModal open={addOpen} onClose={() => { addOpen = false; }} />
{:else}
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">
    <header class="space-y-3">
      <div class="flex items-center justify-between gap-3">
        <div>
          <h1 class="text-lg font-semibold text-text">Thư viện</h1>
          <p class="text-xs text-text-muted">{works.length} truyện</p>
        </div>
        <div class="flex items-center gap-2">
          <button
            type="button"
            onclick={refreshUpdates}
            disabled={checking}
            aria-label="Kiểm tra chương mới"
            class="inline-flex items-center justify-center gap-1.5 h-7 px-2.5 rounded-sm bg-transparent text-text-muted hover:text-text hover:bg-hover text-xs font-medium transition-colors disabled:opacity-60 disabled:cursor-wait cursor-pointer"
          >
            <RotateCw size={13} class={checking ? 'animate-spin' : ''} /> {checking ? 'Đang kiểm tra…' : 'Cập nhật'}
          </button>
          <Button variant="secondary" size="sm" onclick={() => { addOpen = true; }}>
            <Plus size={14} /> Thêm
          </Button>
          <a href="/explore" class="inline-flex items-center justify-center gap-1.5 h-7 px-2.5 rounded-sm bg-transparent text-text-muted hover:text-text hover:bg-hover text-xs font-medium transition-colors">
            Khám phá
          </a>
        </div>
      </div>

      <div class="flex flex-wrap gap-2">
        {#each statusTabs as tab (tab.value)}
          {@const active = status === tab.value}
          <button
            type="button"
            onclick={() => { status = tab.value; }}
            class={cn(
              'inline-flex items-center gap-1.5 h-7 px-3 rounded-full text-xs font-medium transition-colors',
              active ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {tab.label}
            <span class="tabular-nums text-[10px] opacity-80">{counts[tab.value]}</span>
          </button>
        {/each}
      </div>

      <div class="flex items-center gap-2">
        <div class="relative flex-1 max-w-xl">
          <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
          <input
            type="search"
            bind:value={query}
            placeholder="Tìm trong thư viện…"
            class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors"
          />
        </div>
        <label class="shrink-0">
          <span class="sr-only">Sắp xếp</span>
          <select
            bind:value={sort}
            class="h-8 rounded-sm bg-surface-2 border border-transparent px-2 text-sm text-text-muted hover:bg-hover focus:border-accent focus:outline-none transition-colors cursor-pointer"
          >
            {#each sortOptions as option (option.value)}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        </label>
      </div>
    </header>

    {#if filtered.length === 0}
      <EmptyState title={query ? 'Không tìm thấy' : 'Trống ở mục này'} hint={query ? 'Thử từ khoá khác.' : 'Chuyển sang mục khác hoặc thêm truyện.'} />
    {:else}
      <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-x-3 gap-y-4 sm:gap-x-4 sm:gap-y-5">
        {#each filtered as work (work.id)}
          <WorkCard
            work={{
              id: work.id,
              title: work.title,
              cover_url: work.cover_url,
              source: work.sources[0]?.source ?? null,
              chapter: updates[work.id]?.latest_label ?? null,
              nsfw: work.nsfw,
            }}
            hasUpdate={hasUpdate(work)}
          />
        {/each}
      </div>
    {/if}

    <AddMangaModal open={addOpen} onClose={() => { addOpen = false; }} />
  </div>
{/if}
