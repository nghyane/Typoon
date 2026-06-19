<script lang="ts">
  let {
    label,
    hint,
    children,
  }: {
    label: string;
    hint?: string;
    children: import('svelte').Snippet;
  } = $props();
</script>

<label class="block">
  <span class="block text-xs font-medium text-text-muted mb-1.5">{label}</span>
  {@render children()}
  {#if hint}<span class="block text-xs text-text-subtle mt-1">{hint}</span>{/if}
</label>
