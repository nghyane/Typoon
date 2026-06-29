<script lang="ts">
  import { browser } from '$app/environment';
  import { cn } from '$lib/cn';
  import { Check, Eye } from 'lucide-svelte';
  import { queryClient } from '$lib/queryClient';
  import { getWork } from '$lib/works/repo';
  import { getSource } from '$lib/source/registry';
  import Spinner from './Spinner.svelte';
  import Cover from './Cover.svelte';

  export interface WorkCardData {
    /** Local work id — when present the card links to /w/{id} and warms its query. */
    id?: string;
    title: string;
    cover_url: string | null;
    /** Display label under the title (source id for saved works, source name for browse). */
    source?: string | null;
    /** Latest chapter — shown instead of the source on browse cards where the
     *  source is the same for every card and so repeating it adds no info. */
    chapter?: string | null;
    /** Explicit cover proxy headers; falls back to the source manifest's when omitted. */
    coverHeaders?: Record<string, string> | null;
    badge?: string | null;
    nsfw?: boolean;
    inLibrary?: boolean;
  }

  let {
    work,
    href,
    onclick,
    pending = false,
    disabled = false,
    dimmed = false,
    blurNsfw = false,
    class: cls = '',
  }: {
    work: WorkCardData;
    href?: string;
    /** When set the card is a <button> (e.g. browse cards that create a work first). */
    onclick?: () => void;
    pending?: boolean;
    disabled?: boolean;
    dimmed?: boolean;
    /** Blur 18+ covers (revealed on hover/focus). Opt-in per surface: on for the
     *  home shelves, off on Explore where users are browsing to discover. */
    blurNsfw?: boolean;
    class?: string;
  } = $props();

  const targetHref = $derived(href ?? (work.id ? `/w/${work.id}` : undefined));

  // Touch devices can't hover to un-blur an 18+ cover, so the first tap reveals it
  // and the next opens the work. Desktop keeps the hover-reveal (no extra tap).
  const coarsePointer = browser && window.matchMedia('(hover: none)').matches;
  let revealed = $state(false);
  const blurred = $derived(!!work.nsfw && blurNsfw && !revealed);
  const tapToReveal = $derived(blurred && coarsePointer);
  function handleRevealClick(event: MouseEvent): boolean {
    if (tapToReveal) { event.preventDefault(); revealed = true; return true; }
    return false;
  }

  // `work.source` is a source id for saved works, or already a display name for
  // browse cards. Resolve the manifest so both render the same label + headers.
  const sourceObj = $derived(work.source ? getSource(work.source) : null);
  const sourceLabel = $derived(work.source ? (sourceObj?.manifest.name ?? work.source) : null);

  // Covers from some sources only load with the source's referer/auth headers.
  // Prefer explicit headers (browse), else resolve from the work's source id.
  const coverHeaders = $derived(work.coverHeaders ?? sourceObj?.manifest.imageHeaders);

  // Warm the work-detail query on intent so the next page renders from cache.
  let warmed = false;
  function warm(): void {
    if (warmed || !work.id) return;
    warmed = true;
    const id = work.id;
    queryClient.prefetchQuery({ queryKey: ['work', id], queryFn: () => getWork(id), staleTime: 5 * 60_000 });
  }
</script>

{#snippet inner()}
  <div class="relative aspect-[2/3] rounded-md overflow-hidden bg-surface-2 ring-1 ring-inset ring-white/5 transition duration-200 group-hover:ring-white/15 group-hover:shadow-lg group-hover:shadow-black/40">
    <Cover
      src={work.cover_url}
      headers={coverHeaders}
      title={work.title}
      class={cn(
        'absolute inset-0 transition duration-300 ease-out',
        blurred
          ? 'scale-105 blur-xl group-hover:blur-0 group-focus-visible:blur-0'
          : 'group-hover:scale-105',
      )}
    />
    {#if tapToReveal}
      <span class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 text-text-subtle">
        <Eye size={18} />
        <span class="text-[10px] font-medium">Chạm để hiện</span>
      </span>
    {/if}
    {#if pending}
      <span class="absolute inset-0 grid place-items-center bg-bg/45"><Spinner size={20} /></span>
    {/if}
    {#if work.nsfw}
      <span class="absolute top-1.5 right-1.5 inline-flex items-center h-5 px-1.5 rounded-xs bg-error-bg/90 text-error-text chrome-label font-semibold backdrop-blur-sm">18+</span>
    {/if}
    {#if work.badge}
      <div class="absolute bottom-0 inset-x-0 px-2 pb-1 pt-5 text-xs text-white bg-gradient-to-t from-black/85 to-transparent line-clamp-1">{work.badge}</div>
    {/if}
  </div>
  <div class="px-0.5 min-w-0">
    <div class="text-xs font-medium text-text line-clamp-2 leading-snug transition-colors group-hover:text-accent-text">{work.title}</div>
    {#if work.chapter || sourceLabel}
      <div class="mt-0.5 flex items-center gap-1 text-[10px] uppercase tracking-wide text-text-subtle">
        {#if work.chapter}
          <span class="truncate text-text-muted normal-case tracking-normal">Ch. {work.chapter}</span>
        {:else}
          <span class="truncate">{sourceLabel}</span>
        {/if}
        {#if work.inLibrary}<Check size={11} class="shrink-0 text-success" aria-label="Trong thư viện" />{/if}
      </div>
    {/if}
  </div>
{/snippet}

{#if onclick}
  <button
    type="button"
    onclick={(e) => { if (handleRevealClick(e)) return; onclick?.(); }}
    {disabled}
    aria-busy={pending}
    onpointerenter={warm}
    onfocus={warm}
    ontouchstart={warm}
    class={cn(
      'group flex flex-col gap-2 rounded-md text-left cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed',
      dimmed && 'opacity-50',
      pending && 'cursor-wait',
      cls,
    )}
  >
    {@render inner()}
  </button>
{:else}
  <a
    href={targetHref}
    onclick={handleRevealClick}
    onpointerenter={warm}
    onfocus={warm}
    ontouchstart={warm}
    class={cn(
      'group flex flex-col gap-2 rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
      dimmed && 'opacity-50',
      cls,
    )}
  >
    {@render inner()}
  </a>
{/if}
