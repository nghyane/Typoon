<script lang="ts">
  import { afterNavigate } from '$app/navigation';
  import { Clock, Library } from 'lucide-svelte';
  import { listLibraryWorks, listRecentWorks } from '$lib/works/repo';
  import type { Work } from '$lib/db';
  import WorkShelf from '$lib/ui/WorkShelf.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';

  let recentWorks = $state<Work[]>([]);
  let libraryWorks = $state<Work[]>([]);
  let loading = $state(true);
  let error = $state('');

  async function load(): Promise<void> {
    loading = true;
    error = '';
    try {
      const [recent, library] = await Promise.all([listRecentWorks(15), listLibraryWorks()]);
      recentWorks = recent;
      libraryWorks = library;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  // Refetch on every arrival at home (incl. back from the reader) so "Vừa xem"
  // reflects the latest order — afterNavigate also fires on first mount.
  afterNavigate(() => { void load(); });

  const firstLoad = $derived(loading && recentWorks.length === 0 && libraryWorks.length === 0);
</script>

<svelte:head><title>Hội Mê Truyện</title></svelte:head>

<div class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">
  {#if firstLoad}
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
    <ErrorState title="Không tải được trang chủ" message={error} onRetry={load} retrying={loading} />
  {:else}
    <WorkShelf
      title="Vừa xem"
      icon={Clock}
      works={recentWorks}
      emptyTitle="Chưa mở truyện nào"
      emptyHint="Khám phá truyện để bắt đầu."
      blurNsfw
    />

    <WorkShelf
      title="Thư viện"
      icon={Library}
      works={libraryWorks.slice(0, 20)}
      href="/library"
      emptyTitle="Thư viện trống"
      emptyHint="Lưu truyện để quay lại nhanh hơn."
      blurNsfw
    />
  {/if}
</div>
