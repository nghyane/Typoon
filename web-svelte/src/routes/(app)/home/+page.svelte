<script lang="ts">
  import { Clock, Library } from 'lucide-svelte';
  import { listLibraryWorks, listRecentWorks } from '$lib/works/repo';
  import type { Work } from '$lib/db';
  import WorkShelf from '$lib/ui/WorkShelf.svelte';

  let recentWorks = $state<Work[]>([]);
  let libraryWorks = $state<Work[]>([]);
  let loading = $state(true);
  let error = $state('');

  $effect(() => {
    Promise.all([listRecentWorks(15), listLibraryWorks()])
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
    {#each ['a', 'b'] as key (key)}
      <section class="space-y-3">
        <div class="h-5 w-28 rounded-xs bg-surface-2 animate-pulse"></div>
        <div class="flex gap-3 sm:gap-4 overflow-hidden">
          {#each Array(8) as _, i (i)}
            <div class="shrink-0 basis-[calc((100%_-_1.5rem)/3)] sm:basis-[calc((100%_-_3rem)/4)] md:basis-[calc((100%_-_4rem)/5)] lg:basis-[calc((100%_-_5rem)/6)] flex flex-col gap-2" aria-hidden="true">
              <div class="aspect-[2/3] rounded-md bg-surface-2 animate-pulse"></div>
              <div class="h-3 rounded-xs bg-surface-2 animate-pulse"></div>
            </div>
          {/each}
        </div>
      </section>
    {/each}
  {:else if error}
    <p class="text-error-text text-center py-20">{error}</p>
  {:else}
    <WorkShelf
      title="Vừa xem"
      icon={Clock}
      works={recentWorks}
      emptyTitle="Chưa mở truyện nào"
      emptyHint="Khám phá truyện để bắt đầu."
    />

    <WorkShelf
      title="Thư viện"
      icon={Library}
      works={libraryWorks.slice(0, 20)}
      href="/library"
      emptyTitle="Thư viện trống"
      emptyHint="Lưu truyện để quay lại nhanh hơn."
    />
  {/if}
</div>
