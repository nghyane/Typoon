<script lang="ts">
  import { cn } from '$lib/cn';
  import Cover from './Cover.svelte';

  export interface WorkCardData {
    id: string;
    title: string;
    cover_url: string | null;
    source?: string | null;
    badge?: string | null;
    nsfw?: boolean;
  }

  let { work, class: cls = '' }: { work: WorkCardData; class?: string } = $props();
</script>

<a
  href={`/w/${work.id}`}
  class={cn(
    'group flex flex-col gap-2 rounded-sm overflow-hidden',
    'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
    cls,
  )}
>
  <div class="relative aspect-[3/4] rounded-sm overflow-hidden bg-surface-2">
    <Cover src={work.cover_url} title={work.title} class="absolute inset-0 transition-transform group-hover:scale-[1.02]" />
    {#if work.nsfw}
      <span class="absolute top-1.5 right-1.5 inline-flex items-center h-5 px-1.5 rounded-xs bg-error-bg text-error-text text-xs font-semibold">18+</span>
    {/if}
    {#if work.badge}
      <div class="absolute bottom-0 inset-x-0 px-2 py-1 text-xs text-white bg-bg/90">
        {work.badge}
      </div>
    {/if}
  </div>
  <div class="px-0.5 space-y-0.5">
    <div class="text-xs font-medium text-text line-clamp-2 leading-tight">{work.title}</div>
    {#if work.source}
      <div class="text-xs uppercase tracking-wider text-text-subtle">{work.source}</div>
    {/if}
  </div>
</a>
