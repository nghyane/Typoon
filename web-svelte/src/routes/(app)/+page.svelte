<script lang="ts">
  import { BookOpen, Library } from 'lucide-svelte';
  import { listLibraryWorks, listRecentWorks } from '$lib/works/repo';
  import type { Work } from '$lib/db';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import WorkCard from '$lib/ui/WorkCard.svelte';

  let recentWorks = $state<Work[]>([]);
  let libraryWorks = $state<Work[]>([]);
  let loading = $state(true);
  let error = $state('');

  $effect(() => {
    Promise.all([listRecentWorks(6), listLibraryWorks()])
      .then(([recent, library]) => {
        recentWorks = recent;
        libraryWorks = library;
      })
      .catch((err) => { error = err instanceof Error ? err.message : String(err); })
      .finally(() => { loading = false; });
  });
</script>

<svelte:head><title>Hội Mê Truyện</title></svelte:head>

<div class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">
  {#if loading}
    <div class="py-12 flex justify-center"><Spinner size={20} /></div>
  {:else if error}
    <p class="text-error-text text-center py-20">{error}</p>
  {:else}
    {@render WorkRail({
      title: 'Đọc tiếp',
      icon: BookOpen,
      works: recentWorks,
      emptyTitle: 'Chưa mở truyện nào',
      emptyHint: 'Khám phá truyện để bắt đầu.',
    })}

    {@render WorkRail({
      title: 'Thư viện',
      icon: Library,
      works: libraryWorks.slice(0, 6),
      emptyTitle: 'Thư viện trống',
      emptyHint: 'Lưu truyện để quay lại nhanh hơn.',
      action: true,
    })}
  {/if}
</div>

{#snippet WorkRail({ title, icon: Icon, works, emptyTitle, emptyHint, action = false }: {
  title: string;
  icon: typeof BookOpen;
  works: Work[];
  emptyTitle: string;
  emptyHint: string;
  action?: boolean;
})}
  <section class="space-y-3">
    <div class="flex items-center justify-between gap-3">
      <h2 class="inline-flex items-center gap-2 text-sm font-semibold text-text">
        <Icon size={17} class="text-text-subtle" />
        {title}
      </h2>
      {#if action}
        <a href="/library" class="inline-flex items-center justify-center gap-1.5 h-7 px-2.5 rounded-sm bg-transparent text-text-muted hover:text-text hover:bg-hover text-xs font-medium transition-colors">
          Mở thư viện
        </a>
      {/if}
    </div>

    {#if works.length === 0}
      <div class="rounded-md bg-surface border border-border-soft">
        <EmptyState title={emptyTitle} hint={emptyHint} />
      </div>
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
        {#each works as work (work.id)}
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
  </section>
{/snippet}
