<script lang="ts">
  import { BookOpen, BookmarkPlus, Check, ChevronDown, Plus, Search } from 'lucide-svelte';
  import { addWorkToLibrary, getWork, removeWorkFromLibrary, setWorkLibraryStatus, touchWork } from '$lib/works/repo';
  import { getProgress } from '$lib/works/progress';
  import { cn } from '$lib/cn';
  import { getSource } from '$lib/source/registry';
  import { fetchMangaDetail } from '$lib/source/runtime/endpoints';
  import { stripHtml } from '$lib/string';
  import type { LibraryStatus, Work, WorkSource } from '$lib/db';
  import type { MangaDetail } from '$lib/source/types';
  import Cover from '$lib/ui/Cover.svelte';
  import ChapterRowSkeleton from '$lib/ui/ChapterRowSkeleton.svelte';
  import LinkSearchModal from '$lib/work/LinkSearchModal.svelte';
  import {
    mergeChapters,
    pickBestVersion,
    sortMergedChapters,
    type MergedChapter,
    type SourceVersion,
  } from '$lib/work/chapters';
  import { createQuery, createQueries, createMutation, keepPreviousData, useQueryClient } from '@tanstack/svelte-query';

  let { data } = $props();
  const qc = useQueryClient();

  // ── Work from DB (TanStack query — mutation invalidates) ───────

  const workQuery = createQuery(() => ({
    queryKey: ['work', data.workId] as const,
    queryFn: () => getWork(data.workId),
  }));

  const work = $derived(workQuery.data);

  // Reading progress (refetched on mount, so returning from the reader reflects
  // the chapter just opened). Drives the resume button and the read markers.
  const progressQuery = createQuery(() => ({
    queryKey: ['progress', data.workId] as const,
    queryFn: () => getProgress(data.workId),
  }));
  const readSet = $derived(new Set(progressQuery.data?.read ?? []));
  const lastReadNorm = $derived(progressQuery.data?.last_chapter ?? null);

  let chapterQuery = $state('');
  let newestFirst = $state(true);
  let descOpen = $state(false);
  let attachOpen = $state(false);
  let libraryMenuOpen = $state(false);
  let libraryMenuEl = $state<HTMLDivElement | null>(null);

  const statusOptions: Array<{ code: LibraryStatus; label: string }> = [
    { code: 'reading', label: 'Đang đọc' },
    { code: 'plan', label: 'Để dành' },
    { code: 'done', label: 'Đã đọc' },
  ];
  const statusLabels: Partial<Record<LibraryStatus, string>> = {
    reading: 'Đang đọc',
    plan: 'Để dành',
    done: 'Đã đọc',
  };

  // Touch on load
  $effect(() => { if (work) touchWork(work.id).catch(() => {}); });

  // ── Details from every attached source ─────────────────────────

  const sourceTargets = $derived((work?.sources ?? []).map((origin) => ({
    origin,
    source: getSource(origin.source),
  })));

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
      const detail = detailQueries[index]?.data;
      if (!target.source || !detail) return [];
      return [{ source: target.source, origin: target.origin, refs: detail.chapters }];
    });
  });
  const detailLoading = $derived(detailQueries.some((query) => query.isPending || query.isFetching));
  const detailFailures = $derived(detailQueries.filter((query) => query.error));
  const chapters = $derived(mergeChapters(sourceChapters));
  const detail = $derived(detailQueries.find((query) => query.data)?.data ?? null);
  const coverHeaders = $derived.by(() => {
    if (!work?.cover_url) return detail?.coverHeaders;
    const origin = work.sources.find((source) => source.cover_url === work.cover_url);
    return origin ? sourceImageHeaders(origin.source) : undefined;
  });
  const strippedDescription = $derived(stripHtml(detail?.description ?? ''));
  const descOverflows = $derived(strippedDescription.length > 240);

  const readTarget = $derived(sortMergedChapters(chapters, false)[0] ?? null);
  // Resume from the last-opened chapter when we have one and it still exists in
  // the merged list; otherwise fall back to the first chapter.
  const resumeTarget = $derived(
    lastReadNorm ? chapters.find((chapter) => chapter.numberNorm === lastReadNorm) ?? null : null,
  );
  const visibleRows = $derived(
    sortMergedChapters(chapters, newestFirst)
      .filter((chapter) => {
        const term = chapterQuery.trim().toLowerCase();
        if (!term) return true;
        const versions = chapter.sourceVersions.map((version) => `${version.ref.scanlator ?? ''} ${version.source.manifest.name}`).join(' ');
        return `${chapter.number} ${chapter.label} ${versions}`.toLowerCase().includes(term);
      })
      .map((chapter) => ({
        chapter,
        version: pickBestVersion(chapter, work?.target_lang.toLowerCase() ?? 'vi'),
      })),
  );

  // ── Library toggle (mutation → invalidate work query) ──────────

  // Optimistic library mutations (#2): flip the cached work immediately so the
  // button reflects the new state on tap, then reconcile with the DB on settle
  // and roll back if the write fails.
  const workKey = $derived(['work', data.workId] as const);

  async function optimisticPatch(patch: Partial<Work>): Promise<{ prev: Work | undefined }> {
    await qc.cancelQueries({ queryKey: workKey });
    const prev = qc.getQueryData<Work>(workKey);
    if (prev) qc.setQueryData<Work>(workKey, { ...prev, ...patch });
    return { prev };
  }

  function rollback(ctx: { prev: Work | undefined } | undefined): void {
    if (ctx?.prev) qc.setQueryData<Work>(workKey, ctx.prev);
  }

  // `wasInLibrary` is captured at click time and threaded through as the mutation
  // variable. onMutate runs before mutationFn and optimistically flips the cached
  // `work.in_library`, so mutationFn must NOT re-read `work` to pick the action —
  // by then it's already inverted, and it would call the opposite repo function,
  // leaving the toggle a permanent no-op.
  const toggleMutation = createMutation(() => ({
    mutationFn: (wasInLibrary: boolean) =>
      wasInLibrary ? removeWorkFromLibrary(work!.id) : addWorkToLibrary(work!.id),
    onMutate: (wasInLibrary: boolean) => optimisticPatch({ in_library: !wasInLibrary }),
    onError: (_err, _vars, ctx) => rollback(ctx),
    onSettled: () => qc.invalidateQueries({ queryKey: workKey }),
  }));

  const statusMutation = createMutation(() => ({
    mutationFn: (next: LibraryStatus) => setWorkLibraryStatus(work!.id, next),
    onMutate: (next: LibraryStatus) => optimisticPatch({ library_status: next }),
    onError: (_err, _vars, ctx) => rollback(ctx),
    onSettled: () => qc.invalidateQueries({ queryKey: workKey }),
  }));

  $effect(() => {
    if (!libraryMenuOpen) return;
    const onClick = (event: MouseEvent) => {
      if (libraryMenuEl && !libraryMenuEl.contains(event.target as Node)) libraryMenuOpen = false;
    };
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') libraryMenuOpen = false;
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKeydown);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKeydown);
    };
  });

  function sourceLabel(id: string): string {
    return getSource(id)?.manifest.name ?? id;
  }

  function sourceImageHeaders(id: string): Record<string, string> | undefined {
    return getSource(id)?.manifest.imageHeaders;
  }
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
    <section class="pt-4 pb-3">
      <div class="flex items-start gap-4 sm:gap-5">
        <div class="relative w-[88px] sm:w-28 shrink-0 aspect-[2/3] rounded-md overflow-hidden bg-surface-2">
          <Cover src={work.cover_url ?? detail?.cover ?? null} headers={coverHeaders} title={work.title} class="w-full h-full" />
        </div>
        <div class="flex-1 min-w-0 space-y-2 pt-0.5">
          <h1 class="text-lg sm:text-2xl font-semibold text-text leading-snug line-clamp-3 tracking-tight">{work.title}</h1>
          {#if detail?.author}
            <p class="text-sm text-text-muted truncate -mt-1">{detail.author}</p>
          {/if}
          <div class="flex flex-wrap gap-1.5">
            {#if work.nsfw}<span class="inline-flex items-center h-6 px-2 rounded-xs bg-error-bg text-error-text text-xs font-semibold">18+</span>{/if}
            {#if detail?.status}<span class="inline-flex items-center h-6 px-2 rounded-xs bg-surface-2 text-text-muted text-xs font-semibold">{detail.status}</span>{/if}
          </div>
          <div class="flex items-stretch gap-2 flex-wrap">
            {#if resumeTarget}
              <a href={`/r/${work.id}/${resumeTarget.numberNorm}`} class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-sm font-medium hover:brightness-110 transition-[background-color,color,filter] duration-150">
                <BookOpen size={14} /> Đọc tiếp {resumeTarget.number || resumeTarget.numberNorm}
              </a>
            {:else if readTarget}
              <a href={`/r/${work.id}/${readTarget.numberNorm}`} class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-sm font-medium hover:brightness-110 transition-[background-color,color,filter] duration-150">
                <BookOpen size={14} /> Bắt đầu đọc
              </a>
            {/if}
            {#if work.in_library}
              <div bind:this={libraryMenuEl} class="relative">
                <button type="button" onclick={() => { libraryMenuOpen = !libraryMenuOpen; }}
                  class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-accent/15 text-accent-text text-sm font-medium hover:bg-accent/25 transition-colors cursor-pointer"
                  disabled={toggleMutation.isPending || statusMutation.isPending}
                  aria-haspopup="menu"
                  aria-expanded={libraryMenuOpen}
                >
                  {statusLabels[work.library_status ?? 'reading'] ?? 'Đang đọc'}
                  <ChevronDown size={12} class={libraryMenuOpen ? 'rotate-180 transition-transform' : 'transition-transform'} />
                </button>
                {#if libraryMenuOpen}
                  <div role="menu" class="absolute left-0 top-full mt-1 z-30 min-w-[160px] bg-surface rounded-md border border-border-soft py-1">
                    {#each statusOptions as option (option.code)}
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={work.library_status === option.code}
                        onclick={() => { statusMutation.mutate(option.code); libraryMenuOpen = false; }}
                        class="w-full flex items-center justify-between px-3 py-1.5 text-sm text-left transition-colors cursor-pointer hover:bg-hover text-text-muted hover:text-text"
                      >
                        {option.label}
                        {#if work.library_status === option.code}<span class="text-accent text-xs">✓</span>{/if}
                      </button>
                    {/each}
                    <div class="my-1 border-t border-border-soft"></div>
                    <button type="button" role="menuitem" onclick={() => { toggleMutation.mutate(true); libraryMenuOpen = false; }} class="w-full px-3 py-1.5 text-sm text-left text-error-text hover:bg-hover transition-colors cursor-pointer">
                      Xoá khỏi thư viện
                    </button>
                  </div>
                {/if}
              </div>
            {:else}
              <button type="button" onclick={() => toggleMutation.mutate(false)}
                class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-surface-2 text-text text-sm font-medium hover:bg-interactive-hover transition-colors cursor-pointer"
                disabled={toggleMutation.isPending}
              >
                <BookmarkPlus size={14} /> Thư viện
              </button>
            {/if}
          </div>
        </div>
      </div>
    </section>

    <section class="pt-1 pb-3">
      <div class="flex items-center gap-2 mb-2">
        <h2 class="text-xs uppercase tracking-wider text-text-subtle font-medium">Nguồn</h2>
        <span class="text-xs text-text-subtle tabular-nums">{work.sources.length}</span>
        {#if detailLoading}<span class="ts-spinner-circle size-3 text-text-subtle" aria-label="Đang tải"></span>{/if}
        {#if detailFailures.length > 0}<span class="text-xs text-warning-text">{detailFailures.length} nguồn lỗi</span>{/if}
      </div>
      <div class="flex flex-wrap gap-2">
        {#each work.sources as item, i (`${item.source}:${item.upstream_ref}`)}
          {@render SourceCard({ source: item, isPrimary: i === 0 && work.sources.length > 1 })}
        {/each}
        <button type="button" onclick={() => { attachOpen = true; }} class="flex-1 basis-[200px] min-w-[180px] max-w-[320px] flex items-center gap-2 h-11 px-2 rounded-sm text-sm text-text-muted bg-transparent hover:bg-surface-2 hover:text-text border border-dashed border-border-soft transition-colors cursor-pointer text-left">
          <span class="w-6 h-8 shrink-0 rounded-xs flex items-center justify-center bg-surface-2"><Plus size={12} class="text-text-subtle" /></span>
          Thêm nguồn
        </button>
      </div>
    </section>

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
      <div class="sticky top-0 z-10 pt-3 pb-2 -mx-4 sm:-mx-6 px-4 sm:px-6 bg-bg/95 border-b border-border-soft/60 flex items-center gap-2 flex-wrap sm:flex-nowrap">
        <div class="relative flex-1 min-w-0 sm:max-w-xs order-1">
          <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
          <input type="search" bind:value={chapterQuery} placeholder="Tìm chương…"
            class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors"
          />
        </div>
        <div class="ml-auto flex items-center gap-2 shrink-0 order-4">
          <span class="text-xs text-text-subtle tabular-nums">{visibleRows.length} chương</span>
          <span class="text-text-subtle">·</span>
          <button type="button" onclick={() => { newestFirst = !newestFirst; }}
            class="inline-flex items-center gap-1 text-xs text-text-subtle hover:text-text transition-colors cursor-pointer"
          >{newestFirst ? 'Mới nhất' : 'Cũ nhất'}</button>
        </div>
      </div>

      {#if detailLoading && chapters.length === 0}
        <div class="border-t border-border-soft/60">
          <ChapterRowSkeleton count={8} />
        </div>
      {:else if detailFailures.length > 0 && chapters.length === 0}
        <p class="text-error-text text-sm py-8 text-center">Không tải được chương từ các nguồn đã liên kết.</p>
      {:else if chapters.length === 0}
        <p class="text-text-subtle text-sm py-8 text-center">Chưa có chương nào.</p>
      {:else if visibleRows.length === 0}
        <p class="text-text-subtle text-sm py-8 text-center">Không khớp tìm kiếm.</p>
      {:else}
        <div class="border-t border-border-soft/60">
          {#each visibleRows as row (row.chapter.numberNorm)}
            {@render ChapterRow({ workId: work.id, chapter: row.chapter, version: row.version, targetLang: work.target_lang, isRead: readSet.has(row.chapter.numberNorm), isCurrent: row.chapter.numberNorm === lastReadNorm })}
          {/each}
        </div>
      {/if}
    </section>
  {/if}
</div>

{#snippet ChapterRow({ workId, chapter, version, targetLang, isRead, isCurrent }: { workId: string; chapter: MergedChapter; version: SourceVersion | null; targetLang: string; isRead: boolean; isCurrent: boolean })}
  <a href={`/r/${workId}/${chapter.numberNorm}`} class={cn('flex items-center gap-3 px-2 py-2.5 cursor-pointer group hover:bg-surface-2/70 focus-visible:outline-none focus-visible:bg-hover border-b border-border-soft/60 last:border-b-0', isCurrent && 'bg-accent-bg/40')}>
    <span class={cn('tabular-nums font-medium shrink-0 transition-colors', isCurrent ? 'text-accent-text' : isRead ? 'text-text-subtle group-hover:text-text-muted' : 'text-text-muted group-hover:text-text')}>
      {chapter.number || chapter.numberNorm || '?'}
    </span>
    <span class="shrink-0 text-xs uppercase tabular-nums font-medium text-text-subtle">{version?.lang ?? targetLang}</span>
    <span class="hidden sm:flex sm:flex-1 sm:min-w-0 truncate text-text-muted group-hover:text-text transition-colors">
      {#if version?.ref.scanlator}
        <span class="text-text">@{version.ref.scanlator}</span>
        <span class="text-text-subtle"> · {version.source.manifest.name}</span>
      {:else if version}
        <span>{version.source.manifest.name}</span>
      {:else}
        <span>Chưa có nguồn đọc</span>
      {/if}
    </span>
    <div class="ml-auto sm:ml-0 inline-flex items-center gap-3 shrink-0 text-xs text-text-subtle">
      {#if version?.ref.title}<span class="hidden md:inline truncate max-w-[220px]">{version.ref.title}</span>{/if}
      {#if version?.ref.date}<span class="hidden sm:inline whitespace-nowrap tabular-nums">{version.ref.date.slice(0, 10)}</span>{/if}
      {#if isCurrent}
        <span class="text-accent-text whitespace-nowrap font-medium">Đang đọc</span>
      {:else if isRead}
        <Check size={14} class="text-text-subtle" />
      {/if}
    </div>
  </a>
{/snippet}

{#snippet SourceCard({ source, isPrimary }: { source: WorkSource; isPrimary?: boolean })}
  <div class={isPrimary
    ? 'flex-1 basis-[200px] min-w-[180px] max-w-[320px] flex items-center gap-2 h-11 pl-2 pr-1 rounded-sm bg-surface-2 border-l-2 border-accent'
    : 'flex-1 basis-[200px] min-w-[180px] max-w-[320px] flex items-center gap-2 h-11 pl-2 pr-1 rounded-sm bg-surface-2'}>
    <div class="w-6 h-8 shrink-0 rounded-xs overflow-hidden">
      <Cover src={source.cover_url} headers={sourceImageHeaders(source.source)} title={source.title} class="w-full h-full" fontSize="text-[10px]" />
    </div>
    <div class="flex-1 min-w-0 text-sm truncate">
      {#if source.languages?.[0]}<span class="text-xs text-text-subtle font-medium mr-1.5">{source.languages[0].toUpperCase()}</span>{/if}
      <span class="text-text">{sourceLabel(source.source)}</span>
      {#if isPrimary}<span class="ml-1.5 text-xs text-accent">Chính</span>{/if}
    </div>
  </div>
{/snippet}
