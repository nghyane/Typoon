<script module lang="ts">
  // Per-session cooldown so leaving the same work over and over doesn't re-nag
  // with the "continue reading" toast. Module scope survives reader unmounts.
  const RESUME_TOAST_COOLDOWN_MS = 5 * 60_000;
  const lastResumeToastAt = new Map<string, number>();
</script>

<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { afterNavigate, beforeNavigate } from '$app/navigation';
  import { AlertCircle, ChevronDown, ChevronLeft, Download, Eye, EyeOff, Languages, Loader2 } from 'lucide-svelte';
  import { ChapterPages } from '$lib/chapter.svelte';
  import { getSource } from '$lib/source/registry';
  import { resolvePageUrl } from '$lib/source/runtime/endpoints';
  import { cn } from '$lib/cn';
  import { localSettings, READER_PAGE_WIDTH_MAX, READER_PAGE_WIDTH_MIN } from '$lib/localSettings.svelte';
  import { session } from '$lib/auth/session.svelte';
  import { trackDiscordJoinRequired, trackTranslateClick } from '$lib/analytics/client';
  import type { ReaderData } from '$lib/types';
  import { recordRead } from '$lib/works/progress';
  import { getWork } from '$lib/works/repo';
  import { toast } from '$lib/ui/toast.svelte';
  import { ReaderNavigation } from '$lib/reader/ReaderNavigation.svelte';
  import { ReaderSourceResolver } from '$lib/reader/ReaderSourceResolver.svelte';
  import { SvelteReaderTranslation } from '$lib/reader/translation.svelte';
  import ChapterPicker from '$lib/reader/ChapterPicker.svelte';
  import PageRenderer from '$lib/reader/PageRenderer.svelte';
  import SourcePicker from '$lib/reader/SourcePicker.svelte';

  const DISCORD_INVITE_URL = 'https://discord.gg/zuwqbbdZ';

  let { data }: { data: ReaderData | null } = $props();

  // Persist reading history: opening a chapter makes it the work's resume point
  // and marks it read. Keyed on workId+chapterRef so it fires once per chapter,
  // not on every page scroll or translation tick.
  // Snapshot of the chapter currently on screen. Captured so that when the user
  // leaves the reader entirely (onDestroy) we can offer a "continue reading"
  // toast — chapter→chapter navigation reuses this component, so this only fires
  // the toast on a real exit, not on every page turn.
  let resumePoint:
    | { workId: string; href: string; title: string; chapter: string; coverSrc: string | null; coverHeaders: Record<string, string> | null }
    | null = null;

  // Where the user is navigating to as they leave the reader. Used to suppress
  // the resume toast when they're heading to this work's own page — the resume
  // affordance already lives there, so a toast would be redundant.
  let exitDest: string | null = null;
  beforeNavigate((nav) => { exitDest = nav.to?.url.pathname ?? null; });

  // The work cover, fetched once per work, so the toast can lead with the actual
  // thumbnail (recognizable) instead of a generic icon.
  let workCoverSrc = $state<string | null>(null);
  $effect(() => {
    const workId = data?.workId;
    if (!workId) return;
    let cancelled = false;
    void getWork(workId).then((w) => { if (!cancelled) workCoverSrc = w?.cover_url ?? null; });
    return () => { cancelled = true; };
  });

  $effect(() => {
    const workId = data?.workId;
    const chapterRef = data?.chapterRef;
    if (workId && chapterRef) {
      void recordRead(workId, chapterRef);
      resumePoint = {
        workId,
        href: `/r/${workId}/${chapterRef}`,
        title: data?.workTitle ?? 'Hội Mê Truyện',
        chapter: shortChapterNumber(data?.chapterNumber, chapterRef),
        coverSrc: workCoverSrc,
        coverHeaders: (data?.sourceId ? getSource(data.sourceId)?.manifest.imageHeaders : null) ?? null,
      };
    }
  });

  let chapterPickerOpen = $state(false);
  let sourcePickerOpen = $state(false);
  let stripEl = $state<HTMLDivElement | null>(null);
  let contentOverlayEl = $state<HTMLDivElement | null>(null);
  let chapterTriggerEl = $state<HTMLButtonElement | null>(null);
  let sourceTriggerEl = $state<HTMLButtonElement | null>(null);
  let joinRequiredTracked = $state(false);
  let translationHidden = $state(false);
  let pendingRestoreTop = $state<number | null>(null);

  const translation = new SvelteReaderTranslation();

  const source = new ReaderSourceResolver({
    data: () => data,
    onBeforeSwitch: () => translation.cancel(),
    onAfterSwitch: () => {
      nav.pageIndex = 0; nav.scrollProgress = 0; nav.chromeVisible = true;
      stripEl?.scrollTo({ top: 0, behavior: 'auto' });
    },
  });

  const pages = new ChapterPages(
    () => source.activeUrls,
    async (index, rawUrl, opts) => {
      if (rawUrl && !opts?.refresh) return rawUrl;
      const token = source.activeTokens?.[index];
      const src = source.activeSourceId ? getSource(source.activeSourceId) : null;
      if (!token || !src) return rawUrl;
      return resolvePageUrl(src.manifest, token, {}, opts);
    },
    () => source.activePageHeaders,
  );

  const targetLang = $derived(data?.targetLang ?? localSettings.state.default_target_lang ?? 'vi');
  const canTranslateLanguage = $derived(!sameLanguage(source.activeSourceLang, targetLang));
  // At least one page is translated and on-screen (overlays render incrementally,
  // so this is true mid-translation — not just at phase 'done').
  const hasTranslation = $derived(translation.state.translate.done > 0 || translation.state.phase === 'done');
  const translationVisible = $derived(hasTranslation && !translationHidden);

  $effect(() => {
    const pageCount = source.activeUrls.length;
    if (!data || pageCount <= 0 || !canTranslateLanguage) { translation.clear(); return; }
    translation.setChapter({
      chapterKey: [data.workId, data.chapterRef, source.activeVersionKey ?? '', source.activeSourceId ?? '', pageCount, source.activeSourceLang ?? '', targetLang].join(':'),
      pageCount,
      readPage: (index, signal) => pages.readPage(index, signal),
      pageSize: (index) => pages.pageSizes[index] ?? null,
      sourceLanguage: source.activeSourceLang || null,
      targetLanguage: targetLang,
    });
  });

  $effect(() => {
    data?.workId;
    data?.chapterRef;
    source.activeVersionKey;
    targetLang;
    translationHidden = false;
  });

  const slotCount = $derived(Math.max(source.activeUrls.length, pages.blobs.length));
  const slotIndices = $derived(Array.from({ length: slotCount }, (_, i) => i));
  const fakePageSlots = $derived(source.sourceSwitching && slotCount === 0 ? Array.from({ length: 6 }, (_, i) => i) : []);
  const pageCount = $derived(Math.max(source.activeUrls.length, pages.total, slotCount));

  const nav = new ReaderNavigation(() => pageCount);

  // Preserve reading position across back/forward to the same chapter. The
  // page slots establish their height immediately via contain-intrinsic-size,
  // so restoring scrollTop once any slot exists lands close to where the reader
  // left off, even before the image blobs finish streaming in. A fresh forward
  // navigation never calls restore, so a new chapter still opens at the top.
  export const snapshot = {
    capture: () => ({ scrollTop: stripEl?.scrollTop ?? 0 }),
    restore: (v: { scrollTop: number }): void => { pendingRestoreTop = v.scrollTop; },
  };

  // The reader scrolls inside `stripEl`, not the window. A new chapter must open
  // on the first image; a back/forward move must return to where the reader left
  // off. snapshot.capture/restore (above) is SvelteKit's per-history-entry scroll
  // memory — it only sets `pendingRestoreTop` on a back/forward restore, so a
  // fresh forward move falls through to the top.
  //
  // afterNavigate runs once the navigation settles; it records the target for the
  // chapter we just landed on. The strip itself is briefly unmounted while `data`
  // reloads, so we can't position it here — an effect applies the target once the
  // strip and its first slots exist. overflow-anchor:none on the strip then keeps
  // it put as the remaining images stream in (a single assignment holds).
  let chapterTarget = $state<number | null>(null);

  // afterNavigate fires once the navigation settles; default the new chapter to
  // the first image. A back/forward restore arrives separately via
  // snapshot.restore → `pendingRestoreTop` (whose ordering vs afterNavigate isn't
  // guaranteed) and takes precedence in the effect below.
  afterNavigate(() => {
    chapterTarget = 0;
    nav.pageIndex = 0; nav.scrollProgress = 0; nav.chromeVisible = true;
  });

  // Position the strip once it (re)mounts and its first slots have height. Done
  // synchronously — not via rAF, which is throttled in background tabs — so the
  // jump lands reliably. overflow-anchor:none then holds it as images stream in.
  // A pending back/forward restore wins over the default top, whichever arrives
  // first.
  $effect(() => {
    if (!stripEl || slotCount === 0) return;
    if (pendingRestoreTop != null) {
      stripEl.scrollTop = pendingRestoreTop;
      pendingRestoreTop = null;
      chapterTarget = null;
    } else if (chapterTarget != null) {
      stripEl.scrollTop = chapterTarget;
      chapterTarget = null;
    }
  });

  const readingProgress = $derived(pageCount > 0 ? Math.max(0, Math.min(1, nav.scrollProgress)) : 0);
  const currentPageDisplay = $derived(pageCount > 0 ? Math.min(nav.pageIndex + 1, pageCount) : 0);
  const translationBusy = $derived(isTranslationBusy(translation.state.phase));
  const translationBlocked = $derived(session.state.status === 'authenticated' && session.state.user.is_guild_member === false);
  // First-run model prep — the heavy one-time download + runtime startup. Surfaced
  // as a dedicated slide-up panel (not the tiny toolbar chip) so the user reads it
  // like a game's "loading resources" screen instead of a cryptic "OCR 32%".
  // Only the download raises the panel. `resolving` (cache check) and `initializing`
  // (runtime startup) both finish in a blink — on a machine that already cached the
  // model they'd make the panel pop in and vanish, a jarring flash. Those fast states
  // keep the compact "Chuẩn bị" chip; the panel is reserved for the one thing the user
  // genuinely waits on, the multi-second first-run download.
  const modelPrepActive = $derived(canTranslateLanguage && translation.state.model.state === 'downloading');
  const chapterDisplay = $derived(shortChapterNumber(data?.chapterNumber, data?.chapterRef));
  const readerPageWidth = $derived(localSettings.state.reader_page_width);

  onMount(() => { localSettings.load(); });
  onDestroy(() => {
    pages.destroy();
    translation.dispose();
    if (resumePoint) {
      const { workId, href, title, chapter, coverSrc, coverHeaders } = resumePoint;
      // Only nudge when the visit was a real read (turned a page or scrolled in),
      // the user isn't heading to this work's own page, and we haven't just shown
      // it for this work.
      const engaged = nav.pageIndex >= 1 || nav.scrollProgress > 0.08;
      const toOwnWork = !!exitDest && exitDest.startsWith(`/w/${workId}`);
      const now = Date.now();
      const onCooldown = now - (lastResumeToastAt.get(workId) ?? 0) < RESUME_TOAST_COOLDOWN_MS;
      if (engaged && !toOwnWork && !onCooldown) {
        lastResumeToastAt.set(workId, now);
        toast.show({
          key: 'continue-reading',
          cover: { src: coverSrc, headers: coverHeaders, title },
          title,
          description: `Đang đọc · Ch. ${chapter}`,
          action: { label: 'Đọc tiếp', href },
          duration: 8000,
        });
      }
    }
  });

  $effect(() => {
    const host = contentOverlayEl;
    if (!host) return;
    return translation.registerContentHost(host);
  });

  $effect(() => {
    translation.setHidden(translationHidden);
  });

  $effect(() => {
    translation.setProvider(localSettings.state.translation_provider);
  });

  $effect(() => {
    if (!canTranslateLanguage || !translationBlocked) { joinRequiredTracked = false; return; }
    if (joinRequiredTracked) return;
    joinRequiredTracked = true;
    trackDiscordJoinRequired({
      source_id: source.activeSourceId ?? '',
      source_name: source.activeSourceName ?? '',
      page_count: pageCount,
    });
  });

  function handleTranslationButton(): void {
    chapterPickerOpen = false;
    sourcePickerOpen = false;
    const p = translation.state.phase;
    if (translationBlocked && !hasTranslation) return;
    // Start a fresh run only from a settled, un-translated state.
    if ((p === 'ready' || p === 'error') && !hasTranslation) {
      trackTranslateClick({
        phase: p,
        source_id: source.activeSourceId ?? '',
        source_name: source.activeSourceName ?? '',
        source_lang: source.activeSourceLang ?? '',
        target_lang: targetLang,
        page_count: pageCount,
      });
      translationHidden = false;
      translation.translate();
      return;
    }
    // Once any page is translated, the button is a show/hide toggle — usable
    // mid-translation, not gated on phase 'done'.
    if (hasTranslation) translationHidden = !translationHidden;
  }

  function setPageWidth(value: number): void {
    localSettings.update({ reader_page_width: value });
  }

  function eagerPage(index: number): boolean {
    return index >= nav.pageIndex - 1 && index <= nav.pageIndex + 3;
  }

  // Estimated rendered height for content-visibility skipping. The `auto`
  // keyword lets the browser remember the real size after first render, so this
  // only needs to be a reasonable initial guess to avoid scroll jumps.
  function intrinsicHeight(index: number): number {
    const size = pages.pageSizes[index];
    const ratio = size?.width && size.height ? size.height / size.width : 1.5;
    return Math.round(Math.min(readerPageWidth, stripEl?.clientWidth ?? readerPageWidth) * ratio);
  }

  function isTranslationBusy(phase: string): boolean {
    return phase === 'loading' || phase === 'translating';
  }

  function translationActionLabel(hasContent: boolean, hidden: boolean, phase: string): string {
    if (hasContent) return hidden ? 'Hiện' : 'Ẩn';
    if (isTranslationBusy(phase)) return 'Đang…';
    return 'Dịch';
  }

  function translationStatusLabel(state: typeof translation.state): string {
    if (state.model.state === 'downloading') return state.model.ratio === undefined ? 'Tải model' : `Tải ${Math.round(clamp01(state.model.ratio) * 100)}%`;
    if (state.model.state === 'resolving' || state.model.state === 'initializing') return 'Chuẩn bị';
    if (state.phase === 'loading') return 'Chuẩn bị';
    if (state.phase === 'translating') return `${state.translate.done}/${state.translate.total}`;
    return '';
  }

  function translationDetail(state: typeof translation.state, hidden: boolean): string {
    if (state.model.state === 'downloading') return formatModelBytes(state.model.receivedBytes, state.model.totalBytes);
    if (state.model.state === 'resolving') return 'Đang kiểm tra cache model';
    if (state.model.state === 'initializing') return 'Đang khởi động runtime nhận diện';
    if (state.phase === 'ready') return `${state.sourceLanguage?.toUpperCase() ?? 'AUTO'} → ${state.targetLanguage.toUpperCase()}`;
    if (state.phase === 'loading') return 'Tải font và chuẩn bị vùng đọc';
    if (state.phase === 'translating') return `${state.translate.done}/${state.translate.total} trang`;
    if (state.phase === 'done') return hidden ? 'Bấm để hiện bản dịch' : 'Bấm để ẩn bản dịch';
    if (state.phase === 'error') return state.error || 'Bấm để thử lại';
    return 'Sẵn sàng';
  }

  function translationProgress(state: typeof translation.state): number {
    if (state.model.state === 'downloading' && state.model.ratio !== undefined) return clamp01(state.model.ratio);
    if (state.phase === 'translating' && state.translate.total > 0) return clamp01(state.translate.done / state.translate.total);
    return state.phase === 'loading' ? 0.05 : 0;
  }

  // Human stage names for the first-run loader panel — no jargon, names the actual
  // step the way a game names "Downloading assets…" / "Compiling shaders…".
  function modelStageTitle(model: typeof translation.state.model): string {
    if (model.state === 'downloading') return 'Đang tải gói nhận diện';
    if (model.state === 'initializing') return 'Khởi động bộ xử lý ảnh';
    if (model.state === 'resolving') return 'Kiểm tra dữ liệu đã tải';
    return 'Chuẩn bị công cụ dịch';
  }

  function formatModelBytes(received: number | undefined, total: number | undefined): string {
    if (!received && !total) return 'Đang tải model nhận diện chữ';
    if (!total) return `${formatBytes(received ?? 0)} đã tải`;
    return `${formatBytes(received ?? 0)} / ${formatBytes(total)}`;
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024 * 1024) return `${Math.max(0, bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function clamp01(value: number): number {
    return Math.max(0, Math.min(1, value));
  }

  function sameLanguage(sourceLang: string | null | undefined, target: string | null | undefined): boolean {
    const source = normalizeLang(sourceLang);
    const targetLang = normalizeLang(target);
    return !!source && !!targetLang && source === targetLang;
  }

  function normalizeLang(value: string | null | undefined): string {
    const lang = value?.trim().toLowerCase() ?? '';
    if (lang === 'multi' || lang === 'auto') return '';
    return lang;
  }

  function shortChapterNumber(number: string | null | undefined, fallback: string | null | undefined): string {
    const raw = (number ?? fallback ?? '').trim();
    return raw.match(/\d+(?:\.\d+)?/u)?.[0] ?? raw;
  }
</script>

<svelte:head><title>Đọc — Hội Mê Truyện</title></svelte:head>

{#if !data}
  <div class="fixed inset-0 flex items-center justify-center bg-bg"><Loader2 class="animate-spin text-text" size={24} /></div>
{:else}
  <div class="fixed inset-0 bg-bg overflow-hidden text-text">
    <div class="absolute inset-0 overflow-hidden">
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <!-- overflow-anchor:none — the reader sets scroll position deliberately (top
           of the first image on a new chapter, the saved offset on back/forward).
           With anchoring on, the browser fights that: as content-visibility slots
           resolve their real size against the intrinsic estimate it nudges
           scrollTop to keep some mid-list anchor fixed, drifting us off the top
           and causing jumps while pages stream in. -->
      <div bind:this={stripEl} class="w-full h-full overflow-y-auto overflow-x-hidden bg-bg overscroll-contain [overflow-anchor:none]" onscroll={nav.handleStripScroll} onclick={nav.handleStripTap} onkeydown={() => {}} role="presentation">
        <div class="relative min-h-full" style={`width:100%;max-width:${readerPageWidth}px;margin:0 auto;`}>
          {#each slotIndices as i (i)}
            <div
              class="relative w-full overflow-hidden"
              style={`content-visibility:auto;contain-intrinsic-size:auto ${intrinsicHeight(i)}px`}
            >
              <PageRenderer blob={pages.blobs[i] ?? null} index={i} pageSize={pages.pageSizes[i] ?? null} eager={eagerPage(i)} register={(el) => translation.registerPage(i, el)} />
            </div>
          {/each}
          {#if fakePageSlots.length > 0}
            {#each fakePageSlots as _, i (i)}
              <div class="relative w-full overflow-hidden opacity-70">
                <PageRenderer blob={null} index={i} pageSize={null} eager={i < 2} className="w-full" />
              </div>
            {/each}
          {/if}
          <div bind:this={contentOverlayEl} class="absolute inset-0 overflow-hidden pointer-events-none z-[1]" aria-hidden="true"></div>
        </div>
      </div>
    </div>

    <header class={cn('fixed top-0 inset-x-0 z-40 h-[calc(48px+var(--sait))] bg-bg border-b border-border-soft pt-[var(--sait)] pl-[var(--sail)] pr-[var(--sair)] transition-transform duration-200 ease-out', !nav.chromeVisible && '-translate-y-[calc(100%+var(--sait))]')}>
      <div class="mx-auto flex h-12 w-full max-w-7xl items-center gap-2 px-2 sm:px-4">
        <a href="/w/{data.workId}" class="inline-flex items-center justify-center size-9 shrink-0 rounded-sm text-text-muted hover:text-text hover:bg-hover transition-colors" aria-label="Quay lại"><ChevronLeft size={18} /></a>
        <a href="/w/{data.workId}" class="min-w-0 flex-1 leading-tight hover:text-accent-text transition-colors" title={data.workTitle ?? ''}>
          <span class="block truncate text-sm font-semibold text-text">{data.workTitle ?? 'Hội Mê Truyện'}</span>
          <!-- Mobile only: carries source info since the source chip is desktop-only.
               Chapter is omitted here — the chapter chip already shows it. -->
          {#if source.activeSourceName}
            <span class="sm:hidden block truncate text-[11px] text-text-subtle">{source.activeSourceLang ? source.activeSourceLang.toUpperCase() : '?'} · {source.activeSourceName}</span>
          {/if}
        </a>
        <button bind:this={chapterTriggerEl} type="button" onclick={() => { chapterPickerOpen = !chapterPickerOpen; sourcePickerOpen = false; }} class="h-8 px-3 rounded-sm inline-flex items-center gap-1.5 bg-surface-2 text-xs font-medium text-text hover:bg-hover transition-colors cursor-pointer" aria-label="Danh sách chương" aria-expanded={chapterPickerOpen}>
          <span class="tabular-nums">Ch.{chapterDisplay}</span>
          {#if data.chapterTotal}<span class="hidden md:inline text-text-subtle tabular-nums font-normal">{data.chapterIndex ?? 0}/{data.chapterTotal}</span>{/if}
          <ChevronDown size={12} class="text-text-subtle" />
        </button>
        {#if source.activeSourceName}
          <button bind:this={sourceTriggerEl} type="button" onclick={() => { sourcePickerOpen = !sourcePickerOpen; chapterPickerOpen = false; }} class="hidden sm:inline-flex h-8 px-2 rounded-sm items-center gap-1 bg-surface-2 text-[11px] font-medium text-text hover:bg-hover transition-colors cursor-pointer max-w-[9rem]" aria-label="Chọn nguồn" aria-expanded={sourcePickerOpen}>
            {#if source.sourceSwitching}<Loader2 size={11} class="animate-spin text-text-subtle" />{/if}
            {#if source.activeSourceLang}<span class="uppercase tabular-nums text-text-subtle">{source.activeSourceLang}</span>{/if}
            <span class="max-w-[5.5rem] truncate">{source.activeSourceName}</span>
            <ChevronDown size={12} class="text-text-subtle" />
          </button>
        {/if}
      </div>
      <div class="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 overflow-hidden" aria-hidden="true">
        <div class="h-full bg-accent transition-[width] duration-150 ease-out" style={`width:${readingProgress * 100}%`}></div>
      </div>
    </header>

    <!-- First-run model loader — slides up just above the toolbar, game-style:
         names the current step, shows the download size + %, and reassures it's a
         one-time fetch. Non-modal: the raw pages stay visible behind it. Only the
         heavy states (download + runtime startup) raise it; routine per-chapter
         translating keeps the compact toolbar chip. -->
    {#if modelPrepActive}
      <div
        class={cn(
          'fixed inset-x-0 z-40 flex justify-center px-2 bottom-[calc(3.5rem+max(0.5rem,var(--saib)))] pointer-events-none transition-transform duration-200 ease-out',
          !nav.chromeVisible && 'translate-y-[calc(100%+4rem+var(--saib))]',
        )}
      >
        <div class="pointer-events-auto w-full max-w-2xl rounded-md border border-border-soft bg-surface px-3 py-2.5 shadow-lg">
          <div class="flex items-center gap-2.5">
            <span class="grid size-7 shrink-0 place-items-center rounded-sm bg-info-bg text-info-text">
              {#if translation.state.model.state === 'downloading'}<Download size={15} />{:else}<Loader2 size={15} class="animate-spin" />{/if}
            </span>
            <div class="min-w-0 flex-1">
              <p class="truncate text-xs font-semibold text-text">{modelStageTitle(translation.state.model)}</p>
              {#if translation.state.model.state === 'downloading'}
                <p class="truncate text-[11px] tabular-nums text-text-subtle">{formatModelBytes(translation.state.model.receivedBytes, translation.state.model.totalBytes)}</p>
              {/if}
            </div>
            {#if translation.state.model.state === 'downloading' && translation.state.model.ratio !== undefined}
              <span class="shrink-0 text-xs font-bold tabular-nums text-info-text">{Math.round(clamp01(translation.state.model.ratio) * 100)}%</span>
            {/if}
          </div>
          <div class="mt-2 h-1 overflow-hidden rounded-full bg-surface-2">
            {#if translation.state.model.state === 'downloading' && translation.state.model.ratio !== undefined}
              <div class="h-full rounded-full bg-info transition-[width] duration-200 ease-out" style={`width:${clamp01(translation.state.model.ratio) * 100}%`}></div>
            {:else}
              <div class="reader-indeterminate h-full w-2/5 rounded-full bg-info/70"></div>
            {/if}
          </div>
          {#if translation.state.model.state === 'downloading'}
            <p class="mt-1.5 text-[11px] leading-snug text-text-subtle">Chỉ tải một lần — lần sau mở là dùng được ngay.</p>
          {/if}
        </div>
      </div>
    {/if}

    <footer class={cn('fixed inset-x-0 bottom-0 z-40 pb-[max(0.5rem,var(--saib))] pl-[max(0.5rem,var(--sail))] pr-[max(0.5rem,var(--sair))] flex justify-center pointer-events-none transition-transform duration-200 ease-out', !nav.chromeVisible && 'translate-y-[calc(100%+var(--saib))]')}>
      <div class="pointer-events-auto relative flex w-full max-w-2xl items-center gap-1.5 overflow-hidden rounded-md border border-border-soft bg-surface px-1.5 py-1.5">
        <div class="inline-flex min-w-0 items-center gap-1">
          {@render NavBtn({ ref: data.prevRef, workId: data.workId, label: '‹', aria: 'Chương trước' })}
          <span class="min-w-[3.25rem] px-1 text-center text-xs tabular-nums text-text-muted select-none inline-flex items-baseline justify-center gap-1">
            {#if pageCount > 0}<span class="font-semibold text-text">{currentPageDisplay}</span><span class="text-text-subtle">/</span><span>{pageCount}</span>{:else if source.sourceSwitching}<Loader2 size={13} class="animate-spin text-text-subtle" />{:else}<span class="text-text-subtle">—</span>{/if}
          </span>
          {@render NavBtn({ ref: data.nextRef, workId: data.workId, label: '›', aria: 'Chương sau' })}
        </div>
        <span class="flex-1 sm:hidden" aria-hidden="true"></span>
        <label class="hidden h-9 min-w-0 flex-1 items-center gap-2 px-2 text-xs text-text-muted sm:flex">
          <span class="shrink-0 font-medium">Rộng</span>
          <input
            aria-label="Bề rộng tối đa"
            type="range"
            min={READER_PAGE_WIDTH_MIN}
            max={READER_PAGE_WIDTH_MAX}
            step="20"
            value={readerPageWidth}
            oninput={(event) => setPageWidth(Number(event.currentTarget.value))}
            class="reader-slider min-w-0 flex-1 h-1 appearance-none rounded-full bg-surface-2 cursor-pointer accent-accent"
          />
          <span class="hidden lg:inline shrink-0 tabular-nums text-text-subtle">{readerPageWidth}px</span>
        </label>
        <!-- Translation progress — visible on every breakpoint so the reader can
             see how far the run has gotten (model download, prepare, pages done). -->
        {#if canTranslateLanguage && !modelPrepActive && translationStatusLabel(translation.state)}
          <span class="inline-flex h-9 max-w-[5.5rem] shrink-0 items-center truncate rounded-sm px-2 text-xs tabular-nums text-text-subtle" title={translationDetail(translation.state, translationHidden)}>
            {translationStatusLabel(translation.state)}
          </span>
        {/if}
        {#if canTranslateLanguage}
          <button
            class={cn(
              'inline-flex h-9 min-w-[4.75rem] shrink-0 items-center justify-center gap-1.5 rounded-sm border border-transparent px-3 text-xs font-medium transition-[background-color,color,filter] disabled:opacity-50 disabled:cursor-not-allowed',
              hasTranslation
                ? 'border-border-soft bg-surface-2 text-text hover:bg-hover'
                : translationBusy
                  ? 'bg-info-bg text-info-text'
                  : translation.state.phase === 'error' || translation.state.model.state === 'failed'
                    ? 'bg-error-bg text-error-text hover:bg-error-bg/80'
                    : translation.state.phase === 'ready'
                      ? 'border-accent/40 bg-accent text-accent-fg hover:brightness-110'
                      : 'border-border-soft bg-bg text-text hover:bg-hover',
            )}
            disabled={source.sourceSwitching || pageCount <= 0 || translation.state.phase === 'idle' || (translationBlocked && !hasTranslation) || (translationBusy && !hasTranslation)}
            aria-label={hasTranslation ? (translationHidden ? 'Hiện bản dịch' : 'Ẩn bản dịch') : 'Dịch chương'}
            title={translationBlocked ? 'Tham gia Discord để dùng chức năng dịch.' : translationDetail(translation.state, translationHidden)}
            onclick={handleTranslationButton}
          >
            {#if translationBusy && !hasTranslation}<Loader2 class="shrink-0 animate-spin" size={14} />
            {:else if (translation.state.phase === 'error' || translation.state.model.state === 'failed') && !hasTranslation}<AlertCircle class="shrink-0" size={15} />
            {:else if translationVisible}<EyeOff class="shrink-0" size={15} />
            {:else if hasTranslation}<Eye class="shrink-0" size={15} />
            {:else}<Languages class="shrink-0" size={15} />{/if}
            <span>{translationActionLabel(hasTranslation, translationHidden, translation.state.phase)}</span>
          </button>
        {/if}
        {#if canTranslateLanguage && !modelPrepActive && translationProgress(translation.state) > 0}
          <span class="absolute inset-x-0 bottom-0 h-0.5 bg-bg" aria-hidden="true">
            <span class="block h-full bg-accent transition-[width] duration-200" style={`width:${translationProgress(translation.state) * 100}%`}></span>
          </span>
        {/if}
      </div>
    </footer>

    {#if canTranslateLanguage && translationBlocked}
      <div class="fixed left-1/2 -translate-x-1/2 bottom-16 z-40 flex max-w-[90vw] items-center gap-2 rounded-md border border-warning-text/20 bg-warning-bg px-3 py-2 text-xs text-warning-text shadow-lg">
        <AlertCircle size={14} class="shrink-0" />
        <span>Tham gia Discord để dùng chức năng dịch.</span>
        <a href={DISCORD_INVITE_URL} target="_blank" rel="noreferrer" class="shrink-0 font-semibold underline underline-offset-2">Tham gia</a>
      </div>
    {/if}
    {#if source.sourceError}<div class="fixed left-1/2 -translate-x-1/2 bottom-16 z-40 max-w-[90vw] rounded-md bg-error-bg text-error-text border border-border-soft px-3 py-2 text-xs">{source.sourceError}</div>{/if}

    <ChapterPicker open={chapterPickerOpen} anchor={chapterTriggerEl} onClose={() => { chapterPickerOpen = false; }} chapters={data.chapters ?? []} workId={data.workId} currentRef={data.chapterRef} />
    <SourcePicker open={sourcePickerOpen} anchor={sourceTriggerEl} onClose={() => { sourcePickerOpen = false; }} versions={data.versions ?? []} activeKey={source.activeVersionKey} {targetLang} busy={source.sourceSwitching} onPick={(version) => { void source.switchSource(version); }} />
  </div>
{/if}

{#snippet NavBtn({ ref, workId, label, aria }: { ref?: string | null; workId: string; label: string; aria: string })}
  {#if ref}<a href={`/r/${workId}/${ref}`} class="shrink-0 inline-flex items-center justify-center size-9 rounded-sm text-lg leading-none text-text-muted hover:text-text hover:bg-hover transition-colors" aria-label={aria}>{label}</a>
  {:else}<span class="shrink-0 inline-flex items-center justify-center size-9 rounded-sm text-lg leading-none text-text-subtle opacity-40 cursor-not-allowed" aria-disabled="true">{label}</span>{/if}
{/snippet}
