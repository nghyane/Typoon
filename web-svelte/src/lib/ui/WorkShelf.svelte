<script lang="ts">
  import { ChevronLeft, ChevronRight, Clock } from 'lucide-svelte';
  import emblaCarouselSvelte from 'embla-carousel-svelte';
  import type { EmblaCarouselType, EmblaOptionsType } from 'embla-carousel';
  import { cn } from '$lib/cn';
  import type { Work } from '$lib/db';
  import EmptyState from './EmptyState.svelte';
  import WorkCard from './WorkCard.svelte';

  let { title, icon: Icon, works, emptyTitle, emptyHint, href }: {
    title: string;
    icon: typeof Clock;
    works: Work[];
    emptyTitle: string;
    emptyHint: string;
    href?: string;
  } = $props();

  // Embla gives mouse drag + momentum on desktop (native scroll can't), keeps
  // touch swipe on mobile, and snaps to slides. `slidesToScroll: 'auto'` makes
  // the arrows page by however many cards currently fit.
  const options: EmblaOptionsType = { align: 'start', containScroll: 'trimSnaps', slidesToScroll: 'auto' };

  let embla: EmblaCarouselType | undefined;
  let canPrev = $state(false);
  let canNext = $state(false);

  function sync(): void {
    if (!embla) return;
    canPrev = embla.canScrollPrev();
    canNext = embla.canScrollNext();
  }

  function onInit(event: CustomEvent<EmblaCarouselType>): void {
    embla = event.detail;
    sync();
    embla.on('select', sync).on('reInit', sync);
  }

  function cardData(work: Work) {
    return {
      id: work.id,
      title: work.title,
      cover_url: work.cover_url,
      source: work.sources[0]?.source ?? null,
      nsfw: work.nsfw,
    };
  }

  const arrowBase =
    'absolute top-[42%] z-10 hidden size-8 -translate-y-1/2 items-center justify-center rounded-full border border-border-soft bg-surface/95 text-text shadow-lg backdrop-blur transition-opacity hover:bg-hover sm:flex cursor-pointer';
</script>

<section class="space-y-3">
  <div class="flex items-center justify-between gap-3">
    <h2 class="inline-flex items-center gap-2 text-sm font-semibold text-text">
      <Icon size={17} class="text-text-subtle" />
      {title}
    </h2>
    {#if href && works.length > 0}
      <a href={href} class="inline-flex items-center gap-0.5 text-xs font-medium text-text-muted hover:text-text transition-colors">
        Xem tất cả <ChevronRight size={13} />
      </a>
    {/if}
  </div>

  {#if works.length === 0}
    <div class="rounded-md bg-surface border border-border-soft">
      <EmptyState title={emptyTitle} hint={emptyHint} />
    </div>
  {:else}
    <div class="group/shelf relative">
      <button
        type="button"
        aria-label="Cuộn trái"
        onclick={() => embla?.scrollPrev()}
        class={cn(arrowBase, 'left-1', !canPrev ? 'opacity-0 pointer-events-none' : 'opacity-0 group-hover/shelf:opacity-100')}
      >
        <ChevronLeft size={18} />
      </button>

      <div class="overflow-hidden" use:emblaCarouselSvelte={{ options, plugins: [] }} onemblaInit={onInit}>
        <div class="flex gap-3 sm:gap-4">
          {#each works as work (work.id)}
            <div class="min-w-0 shrink-0 grow-0 basis-[calc((100%_-_1.5rem)/3)] sm:basis-[calc((100%_-_3rem)/4)] md:basis-[calc((100%_-_4rem)/5)] lg:basis-[calc((100%_-_5rem)/6)]">
              <WorkCard work={cardData(work)} />
            </div>
          {/each}
        </div>
      </div>

      <button
        type="button"
        aria-label="Cuộn phải"
        onclick={() => embla?.scrollNext()}
        class={cn(arrowBase, 'right-1', !canNext ? 'opacity-0 pointer-events-none' : 'opacity-0 group-hover/shelf:opacity-100')}
      >
        <ChevronRight size={18} />
      </button>
    </div>
  {/if}
</section>
