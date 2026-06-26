<script lang="ts">
  import { Search, Lock } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import {
    pickBestVersion,
    sortMergedChapters,
    type MergedChapter,
    type SourceVersion,
  } from '$lib/work/chapters';

  let {
    chapters,
    targetLang = 'vi',
    workId,
    loading = false,
    failures = 0,
  }: {
    chapters: MergedChapter[];
    targetLang: string;
    workId: string;
    loading?: boolean;
    failures?: number;
  } = $props();

  let chapterQuery = $state('');
  let langFilter = $state('all');
  let newestFirst = $state(true);

  const tgt = $derived(targetLang.toLowerCase());

  $effect(() => { chapters; langFilter = 'all'; });

  const langData = $derived.by(() => {
    const counts = new Map<string, number>();
    for (const ch of chapters) {
      const seen = new Set<string>();
      for (const v of ch.sourceVersions) {
        if (!v.lang || seen.has(v.lang)) continue; // '' = auto-detect, not a filterable language
        seen.add(v.lang);
        counts.set(v.lang, (counts.get(v.lang) ?? 0) + 1);
      }
    }
    return { langs: [...counts.keys()].sort(), counts };
  });

  const sortedChapters = $derived(sortMergedChapters(chapters, newestFirst));

  const visibleRows = $derived.by(() => {
    const targetMode = langFilter === 'all' || langFilter === tgt;
    const term = chapterQuery.trim().toLowerCase();
    const out: Array<{ chapter: MergedChapter; version: SourceVersion | null; key: string; locked: boolean }> = [];

    for (const chapter of sortedChapters) {
      if (term) {
        const hay = `${chapter.number} ${chapter.label} ${chapter.sourceVersions.map((v) => `${v.ref.scanlator ?? ''} ${v.source.manifest.name}`).join(' ')}`.toLowerCase();
        if (!hay.includes(term)) continue;
      }

      if (targetMode) {
        const v = pickBestVersion(chapter, tgt);
        if (langFilter !== 'all' && (!v || v.lang !== tgt)) continue;
        out.push({ chapter, version: v, key: chapter.numberNorm, locked: !!v?.ref.locked });
      } else {
        for (const v of chapter.sourceVersions) {
          if (v.lang !== langFilter) continue;
          out.push({ chapter, version: v, key: `${chapter.numberNorm}:${v.source.manifest.id}:${v.ref.id}`, locked: !!v.ref.locked });
        }
      }
    }
    return out;
  });

  const nonTargetMode = $derived(langFilter !== 'all' && langFilter !== tgt);
</script>

<div class="sticky top-0 z-10 pt-3 pb-2 -mx-4 sm:-mx-6 px-4 sm:px-6 bg-bg/95 border-b border-border-soft/60 flex flex-col gap-2">
  <div class="flex items-center gap-2 flex-wrap sm:flex-nowrap">
    <div class="relative flex-1 min-w-0 sm:max-w-xs order-1">
      <Search size={14} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
      <input type="search" bind:value={chapterQuery} placeholder="Tìm chương…"
        class="h-8 w-full pl-8 pr-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors" />
    </div>
    <div class="ml-auto flex items-center gap-2 shrink-0 order-4">
      <span class="text-xs text-text-subtle tabular-nums">{visibleRows.length} chương</span>
      <span class="text-text-subtle">·</span>
      <button type="button" onclick={() => { newestFirst = !newestFirst; }}
        class="inline-flex items-center gap-1 text-xs text-text-subtle hover:text-text transition-colors cursor-pointer"
      >{newestFirst ? 'Mới nhất' : 'Cũ nhất'}</button>
    </div>
  </div>
  {#if langData.langs.length > 1}
    <div class="flex flex-wrap gap-1.5 pb-1">
      <button type="button" onclick={() => { langFilter = 'all'; }}
        class={cn('h-6 px-2.5 rounded-full text-xs font-medium transition-colors', langFilter === 'all' ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:text-text hover:bg-hover')}
      >Tất cả</button>
      {#each langData.langs as lang (lang)}
        <button type="button" onclick={() => { langFilter = lang; }}
          class={cn('h-6 px-2.5 rounded-full text-xs font-medium uppercase transition-colors', langFilter === lang ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:text-text hover:bg-hover')}
        >{lang} <span class="tabular-nums opacity-60">({langData.counts.get(lang) ?? 0})</span></button>
      {/each}
    </div>
  {/if}
</div>

{#if loading && chapters.length === 0}
  <p class="text-text-subtle text-sm py-8 text-center">Đang tải chương…</p>
{:else if failures > 0 && chapters.length === 0}
  <p class="text-error-text text-sm py-8 text-center">Không tải được chương từ các nguồn đã liên kết.</p>
{:else if chapters.length === 0}
  <p class="text-text-subtle text-sm py-8 text-center">Chưa có chương nào.</p>
{:else if visibleRows.length === 0}
  <p class="text-text-subtle text-sm py-8 text-center">Không khớp tìm kiếm.</p>
{:else}
  <div class="border-t border-border-soft/60">
    {#each visibleRows as row (row.key)}
      {#if row.locked}
        <div aria-disabled="true" title="Chương premium — cần mở khoá trên nguồn"
          class="flex items-center gap-3 px-2 py-2.5 cursor-not-allowed select-none opacity-55 border-b border-border-soft/60 last:border-b-0">
          {@render rowBody(row)}
        </div>
      {:else}
        <a href={`/r/${workId}/${row.chapter.numberNorm}`}
          class="flex items-center gap-3 px-2 py-2.5 cursor-pointer group hover:bg-surface-2/70 focus-visible:outline-none focus-visible:bg-hover border-b border-border-soft/60 last:border-b-0">
          {@render rowBody(row)}
        </a>
      {/if}
    {/each}
  </div>
{/if}

{#snippet rowBody(row: { chapter: MergedChapter; version: SourceVersion | null; locked: boolean })}
  <span class="tabular-nums font-medium text-text-muted group-hover:text-text shrink-0 transition-colors">
    {row.chapter.number || row.chapter.numberNorm || '?'}
  </span>
  <span class="shrink-0 text-xs uppercase tabular-nums font-medium text-text-subtle">{row.version ? (row.version.lang || 'unknown') : tgt}</span>
  {#if row.locked}
    <span class="shrink-0 inline-flex items-center gap-0.5 h-4 px-1 rounded-xs bg-surface-2 text-[10px] font-medium uppercase text-text-subtle">
      <Lock size={10} /> Premium
    </span>
  {/if}
  {#if !nonTargetMode && row.chapter.sourceVersions.length > 1}
    <span class="shrink-0 inline-flex items-center h-4 px-1 rounded-xs bg-surface-2 text-[10px] tabular-nums text-text-subtle">+{row.chapter.sourceVersions.length - 1}</span>
  {/if}
  <span class="hidden sm:flex sm:flex-1 sm:min-w-0 truncate text-text-muted group-hover:text-text transition-colors">
    {#if row.version?.ref.scanlator}
      <span class="text-text">@{row.version.ref.scanlator}</span>
      <span class="text-text-subtle"> · {row.version.source.manifest.name}</span>
    {:else if row.version}
      <span>{row.version.source.manifest.name}</span>
    {:else}
      <span>Chưa có nguồn đọc</span>
    {/if}
  </span>
  <div class="ml-auto sm:ml-0 inline-flex items-center gap-3 shrink-0 text-xs text-text-subtle">
    {#if row.version?.ref.title}<span class="hidden md:inline truncate max-w-[220px]">{row.version.ref.title}</span>{/if}
    {#if row.version?.ref.date}<span class="hidden sm:inline whitespace-nowrap tabular-nums" title={row.version.ref.date.slice(0, 10)}>{row.version.ref.date.slice(0, 10)}</span>{/if}
  </div>
{/snippet}
