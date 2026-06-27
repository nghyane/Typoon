<script lang="ts">
  // Placeholder chapter rows for the work-detail list while source details load,
  // matching ChapterRow's height/padding so the list doesn't jump when real rows
  // arrive. Render inside the bordered list wrapper the real rows use.
  let { count = 8 }: { count?: number } = $props();
</script>

{#each Array(count) as _, i (i)}
  <div class="flex items-center gap-3 px-2 py-2.5 border-b border-border-soft/60 last:border-b-0" aria-hidden="true">
    <div class="h-4 w-8 rounded-xs bg-surface-2 animate-pulse"></div>
    <div class="h-4 w-6 rounded-xs bg-surface-2 animate-pulse"></div>
    <div class="hidden sm:block h-4 flex-1 max-w-[40%] rounded-xs bg-surface-2 animate-pulse"></div>
    <div class="ml-auto h-4 w-16 rounded-xs bg-surface-2 animate-pulse"></div>
  </div>
{/each}
