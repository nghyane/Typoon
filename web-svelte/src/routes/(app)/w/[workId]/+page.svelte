<script lang="ts">
  import { addWorkToLibrary, detachSource, getWork, getWorkHistory, removeWorkFromLibrary, renameWork, setWorkLibraryStatus, touchWork } from '$lib/works/repo';
  import { getSource } from '$lib/source/registry';
  import { fetchMangaDetail } from '$lib/source/runtime/endpoints';
  import { normalizeStatus, stripHtml } from '$lib/string';
  import { localSettings } from '$lib/localSettings.svelte';
  import type { LibraryStatus } from '$lib/db';
  import type { MangaDetail } from '$lib/source/types';
  import { mergeChapters, pickReadTarget } from '$lib/work/chapters';
  import WorkHero from '$lib/work/WorkHero.svelte';
  import WorkSources from '$lib/work/WorkSources.svelte';
  import ChapterList from '$lib/work/ChapterList.svelte';
  import LinkSearchModal from '$lib/work/LinkSearchModal.svelte';
  import { createQuery, createQueries, createMutation, keepPreviousData, useQueryClient } from '@tanstack/svelte-query';

  let { data } = $props();
  const qc = useQueryClient();

  // ── Queries ────────────────────────────────────────────────────

  const workQuery = createQuery(() => ({
    queryKey: ['work', data.workId] as const,
    queryFn: () => getWork(data.workId),
  }));

  const historyQuery = createQuery(() => ({
    queryKey: ['history', data.workId] as const,
    queryFn: () => getWorkHistory(data.workId),
    staleTime: 0,
  }));

  // ── Derived core data ──────────────────────────────────────────

  const work = $derived(workQuery.data);
  const history = $derived(historyQuery.data ?? []);

  const sourceTargets = $derived((work?.sources ?? []).map((origin) => ({
    origin,
    source: getSource(origin.source),
  })));

  const sourceMap = $derived(new Map(
    (work?.sources ?? []).map((s) => [s.source, getSource(s.source)]),
  ));

  const detailQueries = createQueries(() => ({
    queries: sourceTargets.map((target) => ({
      queryKey: ['manga-detail', target.origin.source, target.origin.upstream_ref] as const,
      queryFn: async (): Promise<MangaDetail> => {
        if (!target.source) throw new Error(`Nguồn ${target.origin.source} không khả dụng.`);
        return fetchMangaDetail(target.source.manifest, target.origin.upstream_ref);
      },
      enabled: !!work,
      staleTime: 5 * 60_000,
      retry: false,
      placeholderData: keepPreviousData,
    })),
  }));

  const sourceChapters = $derived.by(() => {
    return sourceTargets.flatMap((target, index) => {
      const d = detailQueries[index]?.data;
      if (!target.source || !d) return [];
      return [{ source: target.source, origin: target.origin, refs: d.chapters }];
    });
  });

  const detailLoading = $derived(detailQueries.some((q) => q.isPending || q.isFetching));
  const detailFailures = $derived(detailQueries.filter((q) => q.error).length);
  const detail = $derived(detailQueries.find((q) => q.data)?.data ?? null);
  const targetLang = $derived(localSettings.state.default_target_lang);
  const chapters = $derived(mergeChapters(sourceChapters, targetLang.toLowerCase()));
  const readTarget = $derived(pickReadTarget(history, chapters));

  const coverHeaders = $derived.by(() => {
    if (!work?.cover_url) return detail?.coverHeaders;
    const origin = work.sources.find((s) => s.cover_url === work.cover_url);
    return origin ? sourceMap.get(origin.source)?.manifest.imageHeaders : undefined;
  });

  const statusLabel = $derived(normalizeStatus(detail?.status));
  const strippedDescription = $derived(stripHtml(detail?.description ?? ''));
  const descOverflows = $derived(strippedDescription.length > 240);

  // ── UI state ───────────────────────────────────────────────────

  let descOpen = $state(false);
  let attachOpen = $state(false);

  // ── Mutations ──────────────────────────────────────────────────

  function invalidateWorkLists() {
    return Promise.all([
      qc.invalidateQueries({ queryKey: ['work', data.workId] }),
      qc.invalidateQueries({ queryKey: ['works', 'library'] }),
      qc.invalidateQueries({ queryKey: ['works', 'recent'] }),
    ]);
  }

  const toggleMutation = createMutation(() => ({
    mutationFn: () =>
      work!.in_library
        ? removeWorkFromLibrary(work!.id)
        : addWorkToLibrary(work!.id),
    onSuccess: invalidateWorkLists,
  }));

  const statusMutation = createMutation(() => ({
    mutationFn: (next: LibraryStatus) => setWorkLibraryStatus(work!.id, next),
    onSuccess: invalidateWorkLists,
  }));

  const detachMutation = createMutation(() => ({
    mutationFn: ({ source, upstreamRef }: { source: string; upstreamRef: string }) =>
      detachSource(work!.id, source, upstreamRef),
    onSuccess: invalidateWorkLists,
  }));

  const renameMutation = createMutation(() => ({
    mutationFn: (title: string) => renameWork(work!.id, title),
    onSuccess: invalidateWorkLists,
  }));

  $effect(() => { if (work) touchWork(work.id).catch(() => {}); });

  const libraryError = $derived(
    (toggleMutation.error ?? statusMutation.error) instanceof Error
      ? ((toggleMutation.error ?? statusMutation.error) as Error).message : '',
  );
  const detachError = $derived(detachMutation.error instanceof Error ? detachMutation.error.message : '');
  const renameError = $derived(renameMutation.error instanceof Error ? renameMutation.error.message : '');
</script>

<svelte:head><title>{work?.title ?? '…'} — Hội Mê Truyện</title></svelte:head>

<div class="max-w-7xl mx-auto px-4 sm:px-6 pb-16">
  {#if workQuery.isPending}
    <div class="flex justify-center py-16"><span class="ts-spinner-circle size-5"></span></div>
  {:else if !work}
    <div class="py-20 text-center">
      <h1 class="text-lg font-semibold text-text">Không tìm thấy truyện</h1>
      <p class="text-sm text-text-muted mt-2">ID: {data.workId}</p>
    </div>
  {:else}
    <WorkHero {work} {detail} {readTarget} {coverHeaders} {statusLabel}
      onToggleLibrary={() => toggleMutation.mutate()}
      onSetStatus={(s) => statusMutation.mutate(s)}
      onRename={(t) => renameMutation.mutate(t)}
      libraryPending={toggleMutation.isPending || statusMutation.isPending}
      {libraryError} {renameError}
    />

    <WorkSources {work} {sourceMap} {detailLoading} {detailFailures}
      onDetach={(source, upstreamRef) => detachMutation.mutate({ source, upstreamRef })}
      onAttach={() => { attachOpen = true; }}
      detachPending={detachMutation.isPending}
      {detachError}
    />

    {#if attachOpen}
      <LinkSearchModal
        open={true}
        onClose={() => { attachOpen = false; }}
        workId={work.id}
        workTitle={work.title}
        ownSources={work.sources}
        onLinked={() => qc.invalidateQueries({ queryKey: ['work', data.workId] })}
      />
    {/if}

    {#if strippedDescription || chapters.length > 0}
      <section class="py-2 space-y-1.5 max-w-3xl">
        {#if strippedDescription}
          <div>
            <p class="text-sm text-text-muted leading-relaxed whitespace-pre-line" class:line-clamp-3={!descOpen}>{strippedDescription}</p>
            {#if descOverflows}
              <button type="button" onclick={() => { descOpen = !descOpen; }}
                class="text-xs text-text-subtle hover:text-text transition-colors cursor-pointer mt-1"
              >{descOpen ? 'Thu gọn' : 'Xem thêm'}</button>
            {/if}
          </div>
        {/if}
        <p class="text-xs text-text-subtle tabular-nums">
          {chapters.length} chương{#if work.updated_at} · {work.updated_at.slice(0, 10)}{/if}
        </p>
      </section>
    {/if}

    <section class="pt-5">
      <ChapterList {chapters} {targetLang} workId={work.id}
        loading={detailLoading} failures={detailFailures} />
    </section>
  {/if}
</div>
