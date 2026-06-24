<script lang="ts">
  import { AlertTriangle, CheckCircle2, ChevronDown, Link as LinkIcon, Power, RotateCw, Search, Wand2 } from 'lucide-svelte';
  import { hasSearch } from '$lib/source/runtime/metadata';
  import { hitKey } from '$lib/source/search';
  import { AddMangaController } from './addManga.svelte';
  import { cn } from '$lib/cn';
  import Button from '$lib/ui/Button.svelte';
  import Cover from '$lib/ui/Cover.svelte';
  import Modal from '$lib/ui/Modal.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';

  let { open, onClose }: { open: boolean; onClose: () => void } = $props();

  const c = new AddMangaController(() => onClose);

  // Own the controller's effects for this component's lifetime.
  $effect(() => c.start());
  // Forward the prop into the controller's reactive state.
  $effect(() => { c.setOpen(open); });

  const inputCls = 'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors';
</script>

<Modal {open} {onClose} title="Thêm manga vào thư viện" size="md">
  <div class="px-5 py-4 space-y-3 min-h-[420px]">
    <div class="relative">
      {#if c.isUrl}
        <LinkIcon size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
      {:else}
        <Search size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
      {/if}
      <input
        type="text"
        value={c.query}
        oninput={(event) => c.setQuery(event.currentTarget.value)}
        disabled={c.busy}
        placeholder="Tìm tên truyện hoặc dán đường dẫn manga"
        class={cn(inputCls, 'pl-9 h-10', c.isUrl && 'pr-36')}
      />
      {#if c.isUrl}
        {@const phase = c.urlPhase}
        {@const ok = phase.kind === 'loading' || phase.kind === 'preview' || phase.kind === 'error'}
        <span class={cn('absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 h-6 px-2 rounded-xs text-xs font-medium pointer-events-none max-w-[8.5rem]', ok ? 'bg-success-bg text-success-text' : 'bg-warning-bg text-warning-text')}>
          {#if ok && 'match' in phase}
            <CheckCircle2 size={12} class="shrink-0" /><span class="truncate">{phase.match.source.manifest.name}</span>
          {:else if phase.kind === 'disabled'}
            <Power size={12} class="shrink-0" /><span class="truncate">Đang tắt</span>
          {:else}
            <AlertTriangle size={12} class="shrink-0" />Chưa hỗ trợ
          {/if}
        </span>
      {/if}
    </div>

    {#if c.isUrl}
      {@const phase = c.urlPhase}
      {#if phase.kind === 'disabled'}
        <div class="rounded-md bg-warning-bg border border-warning-text/20 px-4 py-3">
          <div class="flex items-start gap-3">
            <Power size={14} class="text-warning-text shrink-0 mt-0.5" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-text">Nguồn {phase.match.source.manifest.name} đang tắt</p>
              <p class="text-xs text-text-subtle mt-1 break-all line-clamp-2">{phase.match.upstreamRef}</p>
              <Button variant="secondary" size="sm" onclick={() => c.enableMatchedSource()} class="mt-3">
                <Power size={14} /> Bật nguồn {phase.match.source.manifest.name}
              </Button>
            </div>
          </div>
        </div>
      {:else if phase.kind === 'error'}
        <div class="rounded-md bg-error-bg border border-error-text/20 px-4 py-3">
          <div class="flex items-start gap-3">
            <AlertTriangle size={14} class="text-error-text shrink-0 mt-0.5" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-text">Không tải được từ {phase.match.source.manifest.name}</p>
              <p class="text-xs text-error-text mt-1 line-clamp-2">{phase.message}</p>
              <Button variant="secondary" size="sm" onclick={() => c.retryUrlPreview()} class="mt-3">
                <RotateCw size={14} /> Thử lại
              </Button>
            </div>
          </div>
        </div>
      {:else if phase.kind === 'loading'}
        <div class="rounded-md bg-surface-2 px-4 py-3 flex items-center gap-3">
          <Spinner size={14} class="text-info-text shrink-0" />
          <div class="flex-1 min-w-0">
            <p class="text-sm text-text">Đang tải từ {phase.match.source.manifest.name}…</p>
            <p class="text-xs text-text-subtle truncate mt-1">{phase.match.upstreamRef}</p>
          </div>
        </div>
      {:else if phase.kind === 'preview'}
        <div class="rounded-md bg-surface-2 overflow-hidden">
          <div class="flex items-stretch gap-3 p-3">
            <Cover src={phase.detail.cover} headers={phase.detail.coverHeaders} title={phase.detail.title} class="w-16 aspect-[2/3] rounded-xs shrink-0" fontSize="text-sm" />
            <div class="flex-1 min-w-0 flex flex-col">
              <span class="inline-flex items-center gap-1 text-xs text-success-text font-medium">
                <CheckCircle2 size={12} /> {phase.match.source.manifest.name}
              </span>
              <p class="text-sm font-medium text-text leading-tight mt-1 line-clamp-2">{phase.detail.title}</p>
              {#if phase.detail.author}
                <p class="text-xs text-text-subtle truncate mt-0.5">{phase.detail.author}</p>
              {/if}
              {#if phase.detail.chapters.length > 0}
                <p class="text-xs text-text-subtle mt-auto pt-1">{phase.detail.chapters.length} chương</p>
              {/if}
            </div>
          </div>
          <button
            type="button"
            onclick={() => c.importPreview()}
            disabled={c.busy}
            class="w-full inline-flex items-center justify-center gap-2 h-10 bg-accent text-accent-fg text-sm font-medium hover:brightness-110 transition-[filter] cursor-pointer disabled:opacity-60 disabled:cursor-wait"
          >
            {#if c.isImportingUrl()}<Spinner size={14} /> Đang thêm…{:else}<CheckCircle2 size={14} /> Thêm vào thư viện{/if}
          </button>
        </div>
      {:else if phase.kind === 'unsupported'}
        <div class="rounded-md bg-warning-bg border border-warning-text/20 px-4 py-3">
          <div class="flex items-start gap-3">
            <AlertTriangle size={14} class="text-warning-text shrink-0 mt-0.5" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-text">Không có nguồn quản lý site này</p>
              <p class="text-xs text-text-subtle mt-1 break-all line-clamp-2">{phase.url}</p>
              <Button variant="secondary" size="sm" onclick={() => c.importBlank('')} disabled={c.busy} class="mt-3">
                {#if c.pendingKey === 'blank'}<Spinner size={14} />{:else}<Wand2 size={14} />{/if}
                Tạo trống thay
              </Button>
            </div>
          </div>
        </div>
      {/if}
    {:else if c.debouncedQuery.length < 2}
      {#if c.sources.length === 0}
        <div class="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
          <p class="text-sm text-text-muted">Chưa cài nguồn nào</p>
          <p class="text-xs text-text-subtle mt-1">Mở Cài đặt để cài nguồn đầu tiên.</p>
        </div>
      {:else}
        <div class="space-y-2">
          <p class="text-xs text-text-subtle px-0.5">Bấm để bật/tắt nguồn cho fanout search</p>
          <ul class="flex flex-wrap gap-2">
            {#each c.sources as source (source.manifest.id)}
              {@const searchable = hasSearch(source.manifest)}
              {@const enabled = source.enabled && searchable}
              <li>
                <button
                  type="button"
                  onclick={() => c.toggleSource(source)}
                  disabled={!searchable || c.busy}
                  title={searchable ? (enabled ? `Tắt ${source.manifest.name}` : `Bật ${source.manifest.name}`) : `${source.manifest.name} chưa hỗ trợ tìm — dán link để thêm`}
                  class={cn(
                    'inline-flex items-center gap-2 h-8 pl-2 pr-3 rounded-sm text-xs transition-colors',
                    !searchable
                      ? 'bg-surface-2 text-text-subtle cursor-not-allowed border border-border-soft opacity-50'
                      : enabled
                      ? 'bg-accent-bg text-text border border-accent-text/30 hover:brightness-110 cursor-pointer'
                      : 'bg-surface-2 text-text-muted border border-border-soft hover:bg-hover hover:text-text cursor-pointer',
                  )}
                >
                  <span class={cn('size-1.5 rounded-full shrink-0', enabled ? 'bg-accent' : searchable ? 'bg-text-subtle/40' : 'bg-text-subtle/20')}></span>
                  <span class="font-medium truncate max-w-[140px]">{source.manifest.name}</span>
                  <span class="text-xs text-text-subtle truncate">{source.manifest.host}</span>
                </button>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    {:else}
      {#if c.hits.length > 0 && (c.sourcesWithHits.length > 1 || c.scopeId !== null)}
        <div class="flex items-center gap-1 overflow-x-auto px-0.5" style="scrollbar-width: none">
          <button type="button" onclick={() => c.setScope(null)} class={cn('inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0 transition-colors', c.scopeId === null ? 'bg-surface-2 text-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
            Tất cả <span class={cn('text-xs tabular-nums', c.scopeId === null ? 'text-text-subtle' : 'text-text-subtle/70')}>{c.hits.length}</span>
          </button>
          {#each c.sourcesWithHits as source (source.manifest.id)}
            <button type="button" onclick={() => c.setScope(source.manifest.id)} class={cn('inline-flex items-center gap-2 h-8 px-3 rounded-sm text-sm shrink-0 transition-colors', c.scopeId === source.manifest.id ? 'bg-surface-2 text-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
              {source.manifest.name}
              <span class={cn('text-xs tabular-nums', c.scopeId === source.manifest.id ? 'text-text-subtle' : 'text-text-subtle/70')}>{c.hitCounts.get(source.manifest.id) ?? 0}</span>
            </button>
          {/each}
        </div>
      {/if}

      {#if c.searching && c.scopedHits.length === 0}
        <div class="flex items-center gap-3 px-4 py-3 rounded-md bg-surface-2">
          <Spinner size={14} class="text-info-text" />
          <p class="text-sm text-text-muted">Đang tìm trên {c.visibleSources.length} nguồn…</p>
        </div>
      {:else}
        <div class="space-y-3">
          <div class="flex items-center justify-between gap-2 px-0.5">
            <p class="text-xs uppercase tracking-wider text-text-subtle">
              {c.scopedHits.length} kết quả{#if c.searching}<span class="ml-1.5 normal-case">· đang tìm thêm…</span>{/if}
            </p>
            {#if c.failures.length > 0}
              <span class="text-xs text-warning-text inline-flex items-center gap-1"><AlertTriangle size={12} />{c.failures.length} nguồn lỗi</span>
            {/if}
          </div>

          {#each c.resultGroups as group (group.source.manifest.id)}
            {@const visible = c.visibleHitsFor(group)}
            {@const more = c.moreCountFor(group)}
            <section>
              {#if !c.singleSourceResults}
                <header class="flex items-baseline justify-between gap-2 px-1 mb-1.5">
                  <div class="flex items-baseline gap-2 min-w-0">
                    <span class="text-xs font-medium text-text truncate">{group.source.manifest.name}</span>
                    <span class="text-xs text-text-subtle truncate">{group.source.manifest.host}</span>
                  </div>
                  <span class="text-xs text-text-subtle shrink-0">{group.hits.length}</span>
                </header>
              {/if}
              <ul class="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
                {#each visible as hit (hitKey(hit))}
                  {@const key = hitKey(hit)}
                  <li>
                    <button
                      type="button"
                      onclick={() => c.importHit(hit)}
                      disabled={c.busy}
                      class={cn(
                        'w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-hover transition-colors cursor-pointer',
                        c.pendingKey === key && 'opacity-60 cursor-wait',
                        c.busy && c.pendingKey !== key && 'opacity-60 cursor-not-allowed',
                      )}
                    >
                      <Cover src={hit.manga.cover} headers={hit.manga.coverHeaders} title={hit.manga.title} class="w-8 aspect-[2/3] rounded-xs shrink-0" fontSize="text-xs" />
                      <div class="flex-1 min-w-0">
                        <p class="text-sm text-text truncate leading-tight">{hit.manga.title}</p>
                        {#if hit.source.manifest.languages.length > 0}
                          <p class="text-xs text-text-subtle uppercase mt-1">{hit.source.manifest.languages.slice(0, 3).join('/')}</p>
                        {/if}
                      </div>
                      {#if c.pendingKey === key}<Spinner size={14} class="text-text-subtle shrink-0" />{/if}
                    </button>
                  </li>
                {/each}
                {#if more > 0}
                  <li>
                    <button type="button" onclick={() => c.expandSource(group.source.manifest.id)} class="w-full inline-flex items-center justify-center gap-2 h-8 text-xs text-text-muted hover:bg-hover hover:text-text transition-colors cursor-pointer">
                      <ChevronDown size={12} /> Xem thêm {more}
                    </button>
                  </li>
                {/if}
              </ul>
            </section>
          {/each}
        </div>
      {/if}

      <button
        type="button"
        onclick={() => c.importBlank(c.query)}
        disabled={c.busy}
        class={cn(
          'w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-colors cursor-pointer disabled:cursor-wait disabled:opacity-60',
          c.scopedHits.length === 0 ? 'bg-accent-bg hover:brightness-110' : 'bg-surface-2 hover:bg-hover',
        )}
      >
        <span class={cn('inline-flex items-center justify-center size-8 rounded-sm shrink-0', c.scopedHits.length === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted')}>
          {#if c.pendingKey === 'blank'}<Spinner size={14} />{:else}<Wand2 size={14} />{/if}
        </span>
        <div class="flex-1 min-w-0">
          <p class="text-sm text-text">
            {#if c.scopedHits.length === 0}
              {c.trimmed ? `Không tìm thấy. Tạo "${c.trimmed}" trống?` : 'Tạo manga trống'}
            {:else}
              {c.trimmed ? `Không thấy "${c.trimmed}"? Tạo trống` : 'Tạo manga trống'}
            {/if}
          </p>
          <p class="text-xs text-text-subtle mt-1">Vào trang truyện để liên kết nguồn đọc sau.</p>
        </div>
      </button>
    {/if}

    {#if c.error}
      <p class="text-sm text-error-text text-center">{c.error}</p>
    {/if}
  </div>

  {#snippet footerLeft()}
    {c.sources.length} nguồn đã cài
  {/snippet}

  {#snippet footer()}
    <Button variant="ghost" onclick={onClose} disabled={c.busy}>Huỷ</Button>
  {/snippet}
</Modal>
