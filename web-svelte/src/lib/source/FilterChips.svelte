<!-- FilterChips — renders a source's active filters as a chip row. -->
<script lang="ts">
  import FilterGroup from './FilterGroup.svelte';
  import type { Filter } from './types';

  let { filters, selection, onChange }: {
    filters: Filter[];
    selection: Record<string, string | string[]>;
    onChange: (next: Record<string, string | string[]>) => void;
  } = $props();
</script>

{#each filters as filter (filter.id)}
  <FilterGroup {filter} {selection} {onChange} />
{/each}
