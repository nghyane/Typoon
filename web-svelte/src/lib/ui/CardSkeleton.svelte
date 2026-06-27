<script lang="ts">
  // Placeholder cards that match WorkCard's shape (3:4 cover + two title lines),
  // shown while a grid loads so content fades in place instead of popping in and
  // shoving the layout. Render directly inside the same grid wrapper the real
  // cards use — the component adds no wrapper element, so each item becomes a
  // grid cell.
  let { count = 12 }: { count?: number } = $props();
</script>

{#each Array(count) as _, i (i)}
  <div class="flex flex-col gap-2" aria-hidden="true">
    <div class="aspect-[2/3] rounded-md bg-surface-2 animate-pulse"></div>
    <div class="px-0.5 space-y-1.5">
      <div class="h-3 rounded-xs bg-surface-2 animate-pulse"></div>
      <div class="h-2.5 w-1/2 rounded-xs bg-surface-2 animate-pulse"></div>
    </div>
  </div>
{/each}
