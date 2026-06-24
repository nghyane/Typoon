<script lang="ts">
  import { Plus, Search } from 'lucide-svelte';
  import { listLibraryWorks } from '$lib/works/repo';
  import type { LibraryStatus } from '$lib/db';
  import { cn } from '$lib/cn';
  import AddMangaModal from '$lib/library/AddMangaModal.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import WorkCard from '$lib/ui/WorkCard.svelte';
  import { createQuery } from '@tanstack/svelte-query';

  type StatusFilter = 'all' | LibraryStatus;
  type SortOrder = 'updated' | 'added' | 'alpha';

  const statusTabs: Array<{ value: StatusFilter; label: string }> = [
    { value: 'all', label: 'Tất cả' },
    { value: 'reading', label: 'Đang đọc' },
    { value: 'plan', label: 'Định đọc' },
    { value: 'done', label: 'Đã đọc' },
    { value: 'dropped', label: 'Bỏ' },
  ];

  const sortOptions: Array<{ value: SortOrder; label: string }> = [
    { value: 'updated', label: 'Cập nhật' },
    { value: 'added', label: 'Mới thêm' },
    { value: 'alpha', label: 'A–Z' },
  ];

  let status = $state<StatusFilter>('all');
  let sort = $state<SortOrder>('updated');
  let query = $state('');
  let addOpen = $state(false);

  const worksQuery = createQuery(() => ({
    queryKey: ['works', 'library'] as const,
    queryFn: () => listLibraryWorks(),
  }));

  const works = $derived(worksQuery.data ?? []);
  const loading = $derived(worksQuery.isPending);
  const error = $derived(worksQuery.error?.message ?? '');

  const counts = $derived({
    all: works.length,
    reading: works.filter((w) => w.library_status === 'reading').length,
    plan: works.filter((w) => w.library_status === 'plan').length,
    done: works.filter((w) => w.library_status === 'done').length,
    dropped: works.filter((w) => w.library_status === 'dropped').length,
  });
  const filtered = $derived.by(() => {
    const term = query.trim().toLowerCase();
    const base = works.filter((work) => {
      if (status !== 'all' && work.library_status !== status) return false;
      return !term || work.title.toLowerCase().includes(term);
    });
    if (sort === 'alpha') return [...base].sort((a, b) => a.title.localeCompare(b.title));
    if (sort === 'added') return [...base].sort((a, b) => (b.library_added_at ?? '').localeCompare(a.library_added_at ?? ''));
    return [...base].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  });
</script>

<svelte:head><title>Thư viện — Hội Mê Truyện</title></svelte:head>

{#if loading}
  <div class="flex justify-center py-16"><Spinner size={20} /></div>
{:else if error}
  <p class="text-error-text text-center py-20">{error}</p>
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
        <div class="relative max-w-xl flex-1">
          <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
          <input
            type="search"
            bind:value={query}
            placeholder="Tìm trong thư viện…"
            class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors"
          />
        </div>
        <div class="flex gap-1 shrink-0">
          {#each sortOptions as opt (opt.value)}
            <button
              type="button"
              onclick={() => { sort = opt.value; }}
              class={cn(
                'h-8 px-2.5 rounded-sm text-xs font-medium transition-colors',
                sort === opt.value ? 'bg-surface-2 text-text' : 'text-text-subtle hover:text-text hover:bg-hover',
              )}
            >{opt.label}</button>
          {/each}
        </div>
      </div>
    </header>

    {#if filtered.length === 0}
      <EmptyState title={query ? 'Không tìm thấy' : 'Trống ở mục này'} hint={query ? 'Thử từ khoá khác.' : 'Chuyển sang mục khác hoặc thêm truyện.'} />
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
        {#each filtered as work (work.id)}
          <WorkCard work={{
            id: work.id,
            title: work.title,
            cover_url: work.cover_url,
            source: work.sources[0]?.source ?? null,
            nsfw: work.nsfw,
          }} />
        {/each}
      </div>
    {/if}

    <AddMangaModal open={addOpen} onClose={() => { addOpen = false; }} />
  </div>
{/if}
