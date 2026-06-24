<script lang="ts">
  import { BookOpen, BookmarkPlus, ChevronDown, Pencil } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import type { LibraryStatus, Work } from '$lib/db';
  import type { MangaDetail } from '$lib/source/types';
  import type { ReadTarget } from '$lib/work/chapters';
  import Cover from '$lib/ui/Cover.svelte';

  let {
    work,
    detail = null,
    readTarget = null,
    coverHeaders,
    statusLabel = null,
    libraryError = '',
    renameError = '',
    onToggleLibrary = () => {},
    onSetStatus = (_s: LibraryStatus) => {},
    onRename = (_t: string) => {},
    libraryPending = false,
  }: {
    work: Work;
    detail: MangaDetail | null;
    readTarget: ReadTarget | null;
    coverHeaders: Record<string, string> | undefined;
    statusLabel: string | null;
    libraryError?: string;
    renameError?: string;
    onToggleLibrary: () => void;
    onSetStatus: (s: LibraryStatus) => void;
    onRename: (t: string) => void;
    libraryPending: boolean;
  } = $props();

  let renaming = $state(false);
  let renameValue = $state('');
  let libraryMenuOpen = $state(false);
  let libraryMenuEl = $state<HTMLDivElement | null>(null);

  const STATUS_OPTIONS: Array<{ code: LibraryStatus; label: string }> = [
    { code: 'reading', label: 'Đang đọc' },
    { code: 'plan', label: 'Để dành' },
    { code: 'done', label: 'Đã đọc' },
  ];
  const STATUS_LABELS: Record<string, string> = { reading: 'Đang đọc', plan: 'Để dành', done: 'Đã đọc' };

  $effect(() => {
    if (!libraryMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (libraryMenuEl && !libraryMenuEl.contains(e.target as Node)) libraryMenuOpen = false;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') libraryMenuOpen = false;
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  });

  function startRename(): void {
    renameValue = work.title;
    renaming = true;
  }

  function commitRename(): void {
    const t = renameValue.trim();
    if (t && t !== work.title) onRename(t);
    renaming = false;
  }

  function focusOnMount(el: HTMLElement): void { el.focus(); }
</script>

<section class="pt-4 pb-3">
  <div class="flex items-start gap-4 sm:gap-5">
    <div class="relative w-[88px] sm:w-28 shrink-0 aspect-[2/3] rounded-md overflow-hidden bg-surface-2">
      <Cover src={work.cover_url ?? detail?.cover ?? null} headers={coverHeaders} title={work.title} class="w-full h-full" />
    </div>
    <div class="flex-1 min-w-0 space-y-2 pt-0.5">
      {#if renaming}
        <input type="text"
          bind:value={renameValue}
          onblur={commitRename}
          onkeydown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitRename(); } if (e.key === 'Escape') renaming = false; }}
          maxlength={300}
          use:focusOnMount
          class="w-full text-lg sm:text-2xl font-semibold text-text leading-snug tracking-tight bg-transparent border-b border-border-soft focus:border-accent focus:outline-none pb-0.5" />
      {:else}
        <div class="group/title flex items-start gap-1.5 min-w-0">
          <h1 class="text-lg sm:text-2xl font-semibold text-text leading-snug line-clamp-3 tracking-tight flex-1">{work.title}</h1>
          <button type="button" onclick={startRename} aria-label="Sửa tên"
            class="mt-1 shrink-0 size-7 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover opacity-0 group-hover/title:opacity-100 focus-visible:opacity-100 transition-opacity cursor-pointer">
            <Pencil size={13} />
          </button>
        </div>
      {/if}
      {#if renameError}<p class="text-xs text-error-text mt-0.5">{renameError}</p>{/if}
      {#if detail?.author}
        <p class="text-sm text-text-muted truncate -mt-1">{detail.author}</p>
      {/if}
      <div class="flex flex-wrap gap-1.5">
        {#if work.nsfw}<span class="inline-flex items-center h-6 px-2 rounded-xs bg-error-bg text-error-text text-xs font-semibold">18+</span>{/if}
        {#if statusLabel}<span class="inline-flex items-center h-6 px-2 rounded-xs bg-surface-2 text-text-muted text-xs font-semibold">{statusLabel}</span>{/if}
      </div>
      <div class="flex items-stretch gap-2 flex-wrap">
        {#if readTarget}
          <a href={`/r/${work.id}/${readTarget.ref}`}
            class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-sm font-medium hover:brightness-110 transition-[background-color,color,filter] duration-150">
            <BookOpen size={14} /> {readTarget.isResume ? `Đọc tiếp ch.${readTarget.number}` : 'Bắt đầu đọc'}
          </a>
        {/if}
        {#if work.in_library}
          <div bind:this={libraryMenuEl} class="relative">
            <button type="button" onclick={() => { libraryMenuOpen = !libraryMenuOpen; }}
              class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-accent/15 text-accent-text text-sm font-medium hover:bg-accent/25 transition-colors cursor-pointer"
              disabled={libraryPending}
              aria-haspopup="menu" aria-expanded={libraryMenuOpen}>
              {STATUS_LABELS[work.library_status ?? 'reading'] ?? 'Đang đọc'}
              <ChevronDown size={12} class={libraryMenuOpen ? 'rotate-180 transition-transform' : 'transition-transform'} />
            </button>
            {#if libraryMenuOpen}
              <div role="menu" class="absolute left-0 top-full mt-1 z-30 min-w-[160px] bg-surface rounded-md border border-border-soft py-1">
                {#each STATUS_OPTIONS as opt (opt.code)}
                  <button type="button" role="menuitemradio"
                    aria-checked={work.library_status === opt.code}
                    onclick={() => { onSetStatus(opt.code); libraryMenuOpen = false; }}
                    class="w-full flex items-center justify-between px-3 py-1.5 text-sm text-left transition-colors cursor-pointer hover:bg-hover text-text-muted hover:text-text">
                    {opt.label}
                    {#if work.library_status === opt.code}<span class="text-accent text-xs">✓</span>{/if}
                  </button>
                {/each}
                <div class="my-1 border-t border-border-soft"></div>
                <button type="button" role="menuitem"
                  onclick={() => { onToggleLibrary(); libraryMenuOpen = false; }}
                  class="w-full px-3 py-1.5 text-sm text-left text-error-text hover:bg-hover transition-colors cursor-pointer">
                  Xoá khỏi thư viện
                </button>
              </div>
            {/if}
          </div>
        {:else}
          <button type="button" onclick={onToggleLibrary}
            class="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-sm bg-surface-2 text-text text-sm font-medium hover:bg-interactive-hover transition-colors cursor-pointer"
            disabled={libraryPending}>
            <BookmarkPlus size={14} /> Thư viện
          </button>
        {/if}
        {#if libraryError}<p class="text-xs text-error-text mt-1">{libraryError}</p>{/if}
      </div>
    </div>
  </div>
</section>
